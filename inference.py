import sys, os
from pathlib import Path
import argparse
import numpy as np
import copy
import hashlib
import logging
import math
import random
import time
import datetime
import statistics
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter
from scipy.stats import entropy
from tqdm import tqdm
import wandb
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.visualization import PercentileInterval, ZScaleInterval, MinMaxInterval, BaseInterval
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from tabulate import tabulate
import galsim.roman as roman
from mask import Masker
from unet_model import UNet, UNet_Upsample
from dncnn_model import DnCNN as DnCNN
from utils import utils_image as util
from utils import utils_logger
import zsn2n as ZSN2N
import bm3d
#from skimage.restoration import estimate_sigma

### Variables ###
random_seed = 1024
size_limit = 21e6
plot_size = 7
max_num_imgs_to_show = 5
max_pixel = 65536.0 ## need to set with dataset name to handle natural image denoising
percentile = 99.9
plt.rcParams.update({'font.size': 12})
logger = None
##################################################################################################################
##################################################################################################################
                                        ### Denoiser Classes ###
##################################################################################################################
##################################################################################################################
class BaselineDenoiser():
    def __init__(self):
        self.name = 'Baseline'
        self.scaler = 'noscale'
        self.disable_clipping = False
        self.load()
    def load(self):
        print(self.name)
    def to(self, device):
        pass
    def denoise(self, image):
        return np.expand_dims(np.squeeze(image.cpu().numpy(), axis=1), axis=-1)
    

class UNetDenoiser():
    def __init__(self, model_path, img_channel, device, setting, scaler, dataset_name, train_loss, name, disable_clipping, upsample_mode=True):
        self.img_channel = img_channel
        self.model_path = model_path
        self.name = 'UNet ' + name
        if upsample_mode != None:
            self.model = UNet_Upsample(in_channels=img_channel, out_channels=img_channel, mode=upsample_mode)
        else:
            self.model = UNet(in_channels=img_channel, out_channels=img_channel)
        self.device = device
        self.setting = setting
        self.scaler = scaler
        self.dataset_name = dataset_name
        self.train_loss = train_loss
        self.disable_clipping = disable_clipping
        self.load()
    
    def load(self):
        if self.model_path != 'Random':
            logger.info(self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))
            summary = (self.name, sum(p.numel() for p in self.model.parameters()), self.dataset_name, self.setting, self.scaler, self.train_loss)
            logger.info('%s with %d parameters trained on %s dataset with %s setting, %s scaler, and %s loss loaded!\n'%summary)
        else:
            logger.info('Random UNet Upsample with %d parameters loaded!\n'%sum(p.numel() for p in self.model.parameters()))
        self.model.eval()
        
    def to(self, device):
        self.model = self.model.to(device)
        
    def denoise(self, image):
        denoised_image = self.model(image).detach().cpu().numpy()
        denoised_image = np.expand_dims(np.squeeze(denoised_image, axis=1), axis=-1)
        return denoised_image
    
    
class DnCNNDenoiser():
    def __init__(self, model_path, img_channel, device):
        self.img_channel = img_channel
        self.model_path = model_path
        self.model = DnCNN.DnCNN()
        self.device = device

    def denoise(self, image):
        return image
        
    
class BM3DDenoiser():
    def __init__(self, img_channel):
        self.img_channel = img_channel
        self.VST = False
        self.scaler = 'noscale' if self.VST else 'division'
        self.name = 'BM3D+VST' if self.VST else 'BM3D'
        self.disable_clipping = True
        self.load()
        
    def load(self):
        if self.VST:
            logger.info('BM3D algorithm with VST loaded!\n')
        else:
            logger.info('BM3D algorithm loaded!\n')
        
    def denoise(self, image):
        if self.VST:
            sscaler = StandardScaler()
            sscaler.fit(image.reshape(-1, 1))
            mu, sigma = sscaler.mean_, sscaler.scale_
            #image = sscaler.fit_transform(image.reshape(-1, 1)).reshape(image.shape)
            image /= mu
            noise_level = util.estimate_noise_level(image.squeeze())
            image = util.generalized_anscombe(image, 0, noise_level, 1)
            denoised = bm3d.bm3d(image.astype(np.float32), util.estimate_noise_level(image.squeeze()), 
                                 stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)[:, :, np.newaxis]
            denoised = util.inverse_generalized_anscombe(denoised, 0, noise_level, 1)
            #denoised = denoised*sigma + mu
            denoised *= mu
        else:
            denoised = bm3d.bm3d(image.astype(np.float32), util.estimate_noise_level(image.squeeze()), 
                                 stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)[:, :, np.newaxis]
        return denoised
        
    
class ZSN2NDenoiser():
    def __init__(self, img_channel, device):
        self.name = 'ZSN2N'
        self.max_epoch = 2000
        self.lr = 0.001
        self.step_size = 1500
        self.gamma = 0.5
        self.model = ZSN2N.network(img_channel)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
        self.scaler = 'division'
        self.device = device
        self.disable_clipping = True
        self.load()
        
    def load(self):
        print('Zero-shot Noise2Noise model with %d parameters loaded!\n'%sum(map(lambda x: x.numel(), self.model.parameters())))
        
    def to(self, device):
        self.model = self.model.to(device)
        
    def denoise(self, image):
        #image = torch.from_numpy(image.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(self.device)
        for epoch in range(self.max_epoch):
            ZSN2N.train(self.model, self.optimizer, image)
            self.scheduler.step()
        denoised_image = ZSN2N.denoise(self.model, image)
        denoised_image = denoised_image.detach().cpu().numpy()
        denoised_image = np.expand_dims(np.squeeze(denoised_image, axis=1), axis=-1)
        return denoised_image


class FilterDenoiser():
    def __init__(self, img_channel, mode):
        self.name = mode
        self.img_channel = img_channel
        self.scaler = 'noscale'
        self.disable_clipping = True
        
    def denoise(self, image):
        if self.name=='Median Filter':
            return uniform_filter(image.astype(np.float32), size=7)
        elif self.name=='Mean Filter':
            return median_filter(image.astype(np.float32), size=7)
        elif self.name=='Gaussian Filter':
            return gaussian_filter(image.astype(np.float32), sigma=2)

##################################################################################################################
##################################################################################################################
### Helper Functions ###
##################################################################################################################
##################################################################################################################
# Scaler function that scales data using using one of the 6 scaling methods
def scale(img, scaler):
    if scaler == 'noscale':
        return img, None, None
    elif scaler == 'division':
        return img/65536.0, None, None
    elif scaler == 'norm':
        mmscale = MinMaxInterval()
        a, b = mmscale.get_limits(img)
        return mmscale(img), a, b
    elif scaler == 'standard':
        sscaler = StandardScaler()
        sscaler.fit(img.reshape(-1, 1))
        mu, sigma = sscaler.mean_, sscaler.scale_
        return sscaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape), mu, sigma
    elif scaler == 'arcsinh':
        return np.arcsinh(img), None, None
    elif scaler == 'anscombe':
        gain = roman.gain
        sigma = roman.read_noise + roman.exptime*(roman.dark_current + roman.thermal_backgrounds['Y106'])
        return util.generalized_anscombe(img, 0, sigma, gain), None, None
    

# Descaler function that scales the denoised data back to the original data range
def descale(img, scaler, param1, param2):
    if scaler == 'noscale':
        return img
    elif scaler == 'division':
        return img*65536.0
    elif scaler == 'norm':
        return img*(param2-param1) + param1
    elif scaler == 'standard':
        return img*param2 + param1
    elif scaler == 'arcsinh':
        return np.sinh(img)
    elif scaler == 'anscombe':
        gain = roman.gain
        sigma = roman.read_noise + roman.exptime*(roman.dark_current + roman.thermal_backgrounds['Y106'])
        return util.inverse_generalized_anscombe(img, 0, sigma, gain)

# Function to print the metrics
def summarize_metrics(model_results, metrics_list, aggregate=False):
    model_names = list(model_results.keys())
    headers = ['Model'] + metrics_list
    if aggregate:
        table_data = [[model] + [statistics.mean(metrics[model_name]) for model_name in metrics_list] for model, metrics in model_results.items()]
        logger.info('\n' + tabulate(table_data, headers, tablefmt='fancy_grid'))
    else:
        table_data = [[model] + [metrics[metric][-1] for metric in metrics_list] for model, metrics in model_results.items()]
        logger.info('\n' + tabulate(table_data, headers))

# Function to compute euclidean distance of two pixels
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def calculate_iou(box_a, box_b):
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b
    x_min_inter = max(x_min_a, x_min_b)
    y_min_inter = max(y_min_a, y_min_b)
    x_max_inter = min(x_max_a, x_max_b)
    y_max_inter = min(y_max_a, y_max_b)
    if x_max_inter > x_min_inter and y_max_inter > y_min_inter:
        area_intersection = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        area_intersection = 0
    area_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)
    area_b = (x_max_b - x_min_b) * (y_max_b - y_min_b)
    area_union = area_a + area_b - area_intersection
    if area_union == 0:
        return 0
    return area_intersection / area_union

# Function to compute the different metrics
def calculate_metrics(target, image, objs_X, objs_Y, objs_C, objs_D, border=128, sigma_bkg=3, skip_detection=False, unsupervised=False, source=None):
    pscale = PercentileInterval(percentile)
    if unsupervised:
        mse = util.uMSE(image, target, source)
    else:
        mse = np.mean((target - image)**2)
    mae = np.mean(np.abs(target - image))
    max_pixel = np.max(target) if np.max(target) < 1000.0 else 65536.0
    psnr = 20*math.log10(max_pixel / math.sqrt(mse)) if mse > 0 else 0
    snr = 10*np.log10(np.sum(image**2) / np.sum((image - target)**2))
    ssim = util.calculate_ssim(pscale(target.squeeze()), pscale(image.squeeze()), max_pixel=max_pixel)
    kl = util.kl_divergence(target.ravel(), image.ravel())
    if skip_detection:
        return [psnr, snr, ssim, kl, mse, mae, 0, 0, 0, 0], [], [], [], []
    reference_objs = []
    for x, y, c, d in list(zip(objs_X, objs_Y, objs_C, objs_D)):
        if (x >= border and x <= target.shape[1] - border and y >= border and y <= target.shape[0] - border):
            reference_objs.append((x, y, c, d))
    assignments = [-1] * len(reference_objs)  # -1 means no detection is assigned
    bkg_image = sep.Background(image.squeeze().astype(np.float64))
    bkgsub_image = image.squeeze().astype(np.float64) - bkg_image

    try:
        detected_objs = sep.extract(bkgsub_image, sigma_bkg, err=bkg_image.rms())
    except:
        return [psnr, snr, ssim, kl, mse, mae, 0, 100, len(reference_objs), 0], [], reference_objs, [], assignments
    valid_objs = []
    for idx_detected in range(len(detected_objs)):
        # Check to see if object is not near margins
        if (detected_objs['x'][idx_detected] >= border and detected_objs['x'][idx_detected] <= target.shape[1] - border and
            detected_objs['y'][idx_detected] >= border and detected_objs['y'][idx_detected] <= target.shape[0] - border):
            valid_objs.append(idx_detected)
    detected_objs = detected_objs[valid_objs]
    
    potential_matches = []
    for idx_reference, (x, y, c, d) in enumerate(reference_objs):
        for idx_detected, detection_box in enumerate(detected_objs):
            gt_box = (x - d//2, y - d//2, x + d//2, y + d//2)
            radius = max(detected_objs['a'][idx_detected], detected_objs['b'][idx_detected])
            detection_box = (detected_objs['x'][idx_detected] - radius//2, detected_objs['y'][idx_detected] - radius//2, detected_objs['x'][idx_detected] + radius//2,
                             detected_objs['y'][idx_detected] + radius//2)
            iou = calculate_iou(gt_box, detection_box)
            x_min_a, y_min_a, x_max_a, y_max_a = detection_box
            if d <= 2*(y_max_a - y_min_a)*(x_max_a - x_min_a):
                potential_matches.append((iou, idx_reference, idx_detected))
    potential_matches.sort(reverse=True, key=lambda x:x[0])
    used_detections = set()
    for iou, idx_reference, idx_detected in potential_matches:
        if assignments[idx_reference] == -1 and idx_detected not in used_detections:
            assignments[idx_reference] = idx_detected
            used_detections.add(idx_detected)

    false_alarms = (len(detected_objs) - len(used_detections))*100 / len(detected_objs) if len(detected_objs) else 0
    detection_rate = len([obj for obj in assignments if obj != -1])*100 / len(reference_objs) if len(reference_objs) else 0
    return [psnr, snr, ssim, kl, mse, mae, len(detected_objs), false_alarms, len(reference_objs), detection_rate], detected_objs, reference_objs, used_detections, assignments
    
    
##################################################################################################################
##################################################################################################################
### Main inference script ###
##################################################################################################################
##################################################################################################################
def test(argv):
    #######################
    ### Initializations ###
    #######################
    args = parse_args(argv)
    data_path = args.data_path
    pscale = PercentileInterval(percentile)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result_path = args.result_path + '/' + args.data_path.split('/')[-1] + '_' + str(date) + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    utils_logger.logger_info(result_path, log_path=result_path + 'log.log')
    global logger
    logger = logging.getLogger(result_path)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger.info('Run Identifier: %s' %str(date))
    logger.info('All visualization are done using %.1f%% percentile of images!' % percentile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    denoisers = []
    metrics_total = {}
    if args.bm3d:
        denoisers.append(BM3DDenoiser(args.img_channel)) ### Define classic BM3D denoiser
    if args.filters:
        denoisers.extend([FilterDenoiser(args.img_channel, filter_type) for filter_type in ['Median Filter', 'Mean Filter', 'Gaussian Filter']]) ### Define filtering denoisers
    if args.zsn2n:
        denoiser = ZSN2NDenoiser(args.img_channel, device)
        denoiser.to(device)
        denoisers.append(denoiser) ### Define ZSN2N denoiser
    
    ### Define denoising models
    for idx_model, model_path in enumerate(args.model):
        # Extract model's parameters and settings from the directory name
        if model_path[0] == 'Baseline':
            scaler = 'noscale'
            denoiser = BaselineDenoiser()
        elif model_path[0] == 'Random':
            scaler = 'noscale'
            denoiser = UNetDenoiser(model_path[0], args.img_channel, device, None, scaler, None, None, 'Upsample Random', upsample=True)
        else:
            if model_path[0].split('/')[-1].split('.')[-1] == 'pth':
                complete_model_path = model_path[0]
                configs = model_path[0].split('/')[-2].split('_')
            else:
                complete_model_path = os.path.join(model_path[0], 'best_model.pth')
                configs = model_path[0].split('/')[-1].split('_')
            setting = configs[0]
            if configs[1] in ['xsmall', 'small', 'medium', 'large']:
                dataset_name = '_'.join(configs[1:3])
                ref_index = 3
            else:
                dataset_name = configs[1]
                ref_index = 2
            scaler, train_loss, architecture = configs[ref_index], configs[ref_index+1], configs[ref_index+2]
            if len(configs) >= 8:
                disable_clipping = configs[ref_index+4]=='noclip'
            else:
                disable_clipping = False

            if architecture == 'UNet':
                denoiser = UNetDenoiser(complete_model_path, args.img_channel, device, setting, scaler, dataset_name, train_loss, setting + ' ' + train_loss + ' ' +
                                        str(idx_model + 1), disable_clipping, upsample_mode=None)
            elif 'UNet-Upsample' in architecture:
                upsample_mode = 'bilinear' if architecture=='UNet-Upsample' else architecture[13:]
                denoiser = UNetDenoiser(complete_model_path, args.img_channel, device, setting, scaler, dataset_name, train_loss, 'Upsample ' + setting + ' ' + train_loss + ' ' +
                                        str(idx_model + 1), disable_clipping, upsample_mode=upsample_mode)
            elif architecture == 'DnCNN':
                denoiser = DnCNNDenoiser()
        denoiser.to(device)
        denoisers.append(denoiser)
            
    ### Defining metrics and data path
    metrics_list = ['PSNR', 'SNR', 'SSIM', 'KL Divergence', 'MSE', 'MAE', 'Detection Count', 'False Alarms(%)', 'Reference Count', 'Reference Detected(%)']
    metrics_total['Noisy'] = {metric: [] for metric in metrics_list}
    for denoiser in denoisers:
        metrics_total[denoiser.name] = {metric: [] for metric in metrics_list}
    img_list = [str(file) for file in Path(args.data_path).rglob('*') if (util.is_image_file(str(file)) or util.is_fits_file(str(file)))]

    #################
    ### Inference ###
    #################
    rng = np.random.default_rng(int(hashlib.sha256(data_path.encode()).hexdigest(), 16)%1000) # Generate a random number for each dataset
    height, width, visual_counter = 0, 0, 0
    n_denoisers = len(denoisers)
    for img_name in tqdm(img_list, leave=False, colour='green'):
        ### Reading FITS files
        with fits.open(img_name) as fits_file:
            try:
                img = np.float32(fits_file['SCI'].data)
                header = fits_file['SCI'].header
                if header['RA_V1'] <= 52.9642:
                    continue # Trainng set data
                frame, _, _ = util.read_frame(hf_frame=img, scale_mode=2)
                random_index = random.choice([0, 1])
                other_index = 1 - random_index
                target, _, _ = util.read_frame(hf_frame=frame[random_index:random_index+1,:,:], scale_mode=2, noise_type='None', header=header)
                source, _, _ = util.read_frame(hf_frame=frame[other_index:other_index+1,:,:], scale_mode=2, noise_type='None', header=header)
                objs_X, objs_Y, objs_C, objs_D = [], [], [], []
                unsupervised, is_table_hdu = True, False
                skip_detection = True
                nobjs, exptime = 0, 2748
            except:
                header = fits_file[1].header
                table_data = fits_file[-1].data
                unsupervised, is_table_hdu = False, True
                skip_detection = True if args.noise_type in ['P', 'G', 'PG'] else False
                nobjs, exptime = header['NOBJS'], header['EXPTIME']
                objs_X, objs_Y, objs_C, objs_D = table_data['Object X'], table_data['Object Y'], table_data['Object Type'], table_data['Object Dimension']
                #objs_A, objs_R, objs_IA = table_data['Object Angle'], table_data['Object Ratio'], table_data['Object Initial Angle']
                #objs_HLR1, objs_HLR2, objs_HLR3 = table_data['Object HLR1'], table_data['Object HLR2'], table_data['Object HLR3']
                primary_hdu_idx = 0 if 'NOBJS' in fits_file[0].header else 1
                target, _, _ = util.read_frame(fits_file, primary_hdu_idx)
                if args.noise_type=='None':
                    source, _, _ = util.read_frame(fits_file, primary_hdu_idx + 1, structured_noise=args.structured_noise)
                else:
                    source, gaussian_sigma, poisson_lambda = util.read_frame(fits_file, primary_hdu_idx, noise_type=args.noise_type, structured_noise=args.structured_noise,
                                                                             rng=rng, header=header, subtract_bkg=args.subtract_bkg)

        target, source = np.transpose(target, (1, 2, 0)), np.transpose(source, (1, 2, 0))            
        
        metrics_baseline, detected_objects_baseline, reference_objects_baseline, used_detections_baseline, assignments_baseline = calculate_metrics(target=target, image=source, objs_X=objs_X, objs_Y=objs_Y, objs_C=objs_C, objs_D=objs_D, border=args.overlap, sigma_bkg=args.sigma, skip_detection=False, unsupervised=unsupervised, source=source)
        for metric_name, metric in zip(metrics_list, metrics_baseline):
            metrics_total['Noisy'][metric_name].append(metric)
            
        ##################################################################
        ############## Configure plots for visualization #################
        ##################################################################
        if args.visualize and visual_counter < max_num_imgs_to_show:
            ### Initialize parameters  for visualization
            logger.info('Source file: %s, Exposure time: %ds, Total objects in FOV: %d'%(img_name, exptime, nobjs))
            fits_file = fits.open(img_name)
            if 'JWST' in args.data_path:
                fits_file = fits.HDUList([fits.PrimaryHDU(data=fits_file['SCI'].data[0, :, :]), fits.ImageHDU(data=fits_file['SCI'].data[1, :, :], name='SCI2')])
            elif is_table_hdu:
                fits_file.insert(len(fits_file) - 1, fits.ImageHDU(data=np.squeeze(source.astype(np.float32))))
            else:
                fits_file.append(fits.ImageHDU(data=np.squeeze(source.astype(np.float32))))
            bins = 200
            norm = mcolors.LogNorm(vmin=0.1, vmax=1000.0, clip=True)
            num_subplots = n_denoisers + 2
            num_rows = int((num_subplots - 1) / 3) + 1
            num_cols = 3
            if args.noise_type in ['P', 'G', 'PG']:
                logger.info('Gaussian Noise Parameter: %d, Poisson Noise Paramater:%d'%(gaussian_sigma, poisson_lambda))
                
            ### Plot qualitative result
            fig_qual, axs_qual = plt.subplots(num_rows, num_cols, figsize=(plot_size*num_cols, plot_size*num_rows))
            axs_qual = axs_qual.flatten()
            image_obj = axs_qual[0].imshow(pscale(source), interpolation='nearest', cmap='gray', vmin=0, vmax=1)
            axs_qual[0].set_title('Noisy; PSNR=%.2f'%(metrics_total['Noisy']['PSNR'][-1]))
            cmap, _ = image_obj.get_cmap(), plt.colorbar(image_obj, ax=axs_qual[0], fraction=0.046, pad=0.04)
            axs_qual[len(denoisers) + 1].imshow(pscale(target), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
            axs_qual[len(denoisers) + 1].set_title('Ground Truth')
            
            ### Plot detections on images
            fig_obj, axs_obj = plt.subplots(num_rows, num_cols, figsize=(plot_size*num_cols, plot_size*num_rows))
            axs_obj = axs_obj.flatten()
            bkg_image = sep.Background(source.squeeze().astype(np.float64))
            bkgsub_image = source.squeeze().astype(np.float64) - bkg_image
            image_obj2 = axs_obj[0].imshow(pscale(bkgsub_image), interpolation='nearest', cmap='gray_r', vmin=0, vmax=1)
            axs_obj[0].set_title('Noisy; Ref. Detected%%=%.2f'%(metrics_total['Noisy']['Reference Detected(%)'][-1]))
            cmap2, _ = image_obj2.get_cmap(), plt.colorbar(image_obj2, ax=axs_obj[0], fraction=0.046, pad=0.04)
            
            # Noisy Frame Objects
            for idx_detected in range(len(detected_objects_baseline)):
                e = Ellipse(xy=(detected_objects_baseline['x'][idx_detected], detected_objects_baseline['y'][idx_detected]), width=detected_objects_baseline['a'][idx_detected],
                            height=detected_objects_baseline['b'][idx_detected], angle=detected_objects_baseline['theta'][idx_detected] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                if idx_detected in used_detections_baseline:
                    e.set_edgecolor('green')
                axs_obj[0].add_artist(e)
            for idx_reference, (reference_x, reference_y, _, reference_d) in enumerate(reference_objects_baseline):
                if assignments_baseline[idx_reference] == -1:
                    e = Ellipse(xy=(reference_x, reference_y), width=reference_d, height=reference_d, angle=0)
                    e.set_facecolor('none')
                    e.set_edgecolor('yellow')
                    axs_obj[0].add_artist(e)
            
            # GT Frame Objects
            bkg_image = sep.Background(target.squeeze().astype(np.float64))
            bkgsub_image = target.squeeze().astype(np.float64) - bkg_image
            axs_obj[len(denoisers) + 1].imshow(pscale(bkgsub_image), interpolation='nearest', cmap=cmap2, vmin=0, vmax=1)
            axs_obj[len(denoisers) + 1].set_title('Ground Truth')
            for reference_x, reference_y, _, reference_d in list(reference_objects_baseline):
                e = Ellipse(xy=(reference_x, reference_y), width=reference_d, height=reference_d, angle=0)
                e.set_facecolor('none')
                e.set_edgecolor('green')
                axs_obj[len(denoisers) + 1].add_artist(e)
            
            ### Plot pixel distributions
            fig_dist, axs_dist = plt.subplots(num_rows, num_cols, figsize=(plot_size*num_cols, plot_size*num_rows))
            axs_dist = axs_dist.flatten()
            axs_dist[0].hist(np.ravel(source), bins=bins, histtype='stepfilled')
            axs_dist[0].set_title('Noisy; KL Div.=%.5f'%(metrics_total['Noisy']['KL Divergence'][-1]))
            axs_dist[len(denoisers) + 1].hist(np.ravel(target), bins=bins, histtype='stepfilled')
            axs_dist[len(denoisers) + 1].set_title('Ground Truth')

            ### Plot Error Maps
            num_subplots2 = num_subplots - 1
            num_rows2 = int((num_subplots2 - 1) / 3) + 1
            fig_err, axs_err = plt.subplots(num_rows2, num_cols, figsize=(plot_size*num_cols, plot_size*num_rows2))
            axs_err = axs_err.flatten()
            image_obj = axs_err[0].imshow(np.abs(source - target), interpolation='nearest', cmap='gray_r', norm=norm)
            axs_err[0].set_title('Noisy; MAE=%.3f'%(metrics_total['Noisy']['MAE'][-1]))
            cmap3, _ = image_obj.get_cmap(), plt.colorbar(image_obj, ax=axs_err[0], fraction=0.046, pad=0.04)
            
        
        #####################################################
        ################# Inference #########################
        #####################################################
        ### Extract patches
        scaled_source = util.scale(source, denoiser.scaler)
        if height != target.shape[0] or width != target.shape[1]:
            height, width = target.shape[:2]
            width_list = list(np.arange(0, width, args.patch_size - args.overlap, dtype=np.int_))
            height_list = list(np.arange(0, height, args.patch_size - args.overlap, dtype=np.int_))
            tops, lefts = [], []
            for top in height_list:
                top = (width - args.patch_size) if (top + args.patch_size) >= height else top
                tops.append(top)
            for left in width_list:
                left = (width - args.patch_size) if (left + args.patch_size) >= width else left
                lefts.append(left)
            patch_coordinates = [(top, left) for top in tops for left in lefts]
            
        ### Denoise
        batch_size = 128
        for idx_denoiser in range(len(denoisers)):
            denoiser = denoisers[idx_denoiser]
            denoised_source = np.zeros_like(scaled_source) 
            denoised_source_count = np.zeros_like(scaled_source)
            start_idx = 0
            while start_idx < len(patch_coordinates):
                end_idx = start_idx + batch_size if start_idx + batch_size<=len(patch_coordinates) else len(patch_coordinates)
                scaled_source_patches = torch.permute(torch.stack([torch.from_numpy(scaled_source[top:top + args.patch_size, left:left + args.patch_size,
                                                        :].astype(np.float32)) for top, left in patch_coordinates[start_idx:end_idx]], dim=0), (0, 3, 1, 2))
                with torch.no_grad():
                    estimated = denoiser.denoise(scaled_source_patches.to(device, non_blocking=True))
                    estimated = util.descale(estimated, denoiser.scaler)
                    if 'JWST' in args.data_path or denoiser.disable_clipping:
                        pass
                    else:
                        pass #estimated = np.clip(estimated, 0.0, 65536.0)
                for idx_estimated, (top, left) in enumerate(patch_coordinates[start_idx:end_idx]):
                    denoised_source[top:top + args.patch_size, left:left + args.patch_size, :] += estimated[idx_estimated,:,:,:]
                    denoised_source_count[top:top + args.patch_size, left:left + args.patch_size, :] += 1
                start_idx += batch_size
                
            denoised_source /= denoised_source_count
            metrics, detected_objects_denoiser, reference_objects_denoiser, used_detections_denoiser, assignments_denoiser = calculate_metrics(target, denoised_source, objs_X,
                                                                objs_Y, objs_C, objs_D, args.overlap, args.sigma, skip_detection=skip_detection, unsupervised=unsupervised, source=source)
            for metric_name, metric in zip(metrics_list, metrics):
                metrics_total[denoiser.name][metric_name].append(metric)
                
            ################################################
            ########### Visualize denoised images ##########
            ################################################
            if args.visualize and visual_counter < max_num_imgs_to_show:
                idx_image = idx_denoiser + 1
                ### Visualizing images to compare qualitatively
                axs_qual[idx_image].imshow(pscale(denoised_source), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
                axs_qual[idx_image].set_title('%s; PSNR=%.2f'%(denoiser.name, metrics_total[denoiser.name]['PSNR'][-1]))
                
                ### Show the detected objects in each denoised image
                bkg_image = sep.Background(denoised_source.squeeze().astype(np.float64))
                bkgsub_image = denoised_source.squeeze().astype(np.float64) - bkg_image
                axs_obj[idx_image].imshow(pscale(bkgsub_image), interpolation='nearest', cmap=cmap2, vmin=0, vmax=1)
                axs_obj[idx_image].set_title('%s; Ref. Detected%%=%.2f'%(denoiser.name, metrics_total[denoiser.name]['Reference Detected(%)'][-1]))
                for idx_detected in range(len(detected_objects_denoiser)):
                    e = Ellipse(xy=(detected_objects_denoiser['x'][idx_detected], detected_objects_denoiser['y'][idx_detected]), width=detected_objects_denoiser['a'][idx_detected],
                                height=detected_objects_denoiser['b'][idx_detected], angle=detected_objects_denoiser['theta'][idx_detected] * 180. / np.pi)
                    e.set_facecolor('none')
                    e.set_edgecolor('red')
                    if idx_detected in used_detections_denoiser:
                        e.set_edgecolor('green')
                    axs_obj[idx_image].add_artist(e)
                for idx_reference, (reference_x, reference_y, _, reference_d) in enumerate(reference_objects_denoiser):
                    if assignments_denoiser[idx_reference] == -1:
                        e = Ellipse(xy=(reference_x, reference_y), width=reference_d, height=reference_d, angle=0)
                        e.set_facecolor('none')
                        e.set_edgecolor('yellow')
                        axs_obj[idx_image].add_artist(e)
                    
                ### Distributions
                axs_dist[idx_image].hist(np.ravel(denoised_source), bins=bins, histtype='stepfilled')
                axs_dist[idx_image].set_title('%s; KL Div.=%.5f'%(denoiser.name, metrics_total[denoiser.name]['KL Divergence'][-1]))
                
                ### Error map
                axs_err[idx_image].imshow(np.abs(denoised_source - target), interpolation='nearest', cmap=cmap3, norm=norm)
                axs_err[idx_image].set_title('%s; MAE=%.3f'%(denoiser.name, metrics_total[denoiser.name]['MAE'][-1]))
                
                
                if is_table_hdu:
                    fits_file.insert(len(fits_file) - 1, fits.ImageHDU(data=np.squeeze(denoised_source.astype(np.float32)), name=denoiser.name))
                elif 'JWST' in args.data_path:
                    fits_file.append(fits.ImageHDU(data=np.squeeze(denoised_source.astype(np.float32)), name=denoiser.name))
                else:
                    fits_file.append(fits.ImageHDU(data=np.squeeze(denoised_source.astype(np.float32)), name=denoiser.name))
            
        ### Show the visualizations
        if args.visualize and visual_counter < max_num_imgs_to_show:
            for idx_axs in range(num_subplots, num_rows*num_cols):
                axs_qual[idx_axs].axis('off')
                axs_obj[idx_axs].axis('off')
                axs_dist[idx_axs].axis('off')
            for idx_axs in range(num_subplots):
                if idx_axs!=0:
                    axs_qual[idx_axs].set_xticks([])
                    axs_qual[idx_axs].set_yticks([])
                axs_obj[idx_axs].set_xticks([])
                axs_obj[idx_axs].set_yticks([])
                axs_dist[idx_axs].set_ylim(1e0, 2e7)
                axs_dist[idx_axs].set_yscale('log')
            for idx_axs in range(num_subplots2, num_rows2*num_cols):
                axs_err[idx_axs].axis('off')
            for idx_axs in range(num_subplots2):
                axs_err[idx_axs].set_xticks([])
                axs_err[idx_axs].set_yticks([])    
            plt.tight_layout()
            
            file_path = (result_path + img_name.split('/')[-1])[:-5]
            fits_file.writeto(file_path + '.fits', overwrite=True)
            fits_file.close()
            fig_qual.savefig(file_path + '_visualized.png', dpi=300)
            fig_obj.savefig(file_path + '_objects.png', dpi=300)
            fig_dist.savefig(file_path + '_distributions.png', dpi=300)
            fig_err.savefig(file_path + '_error.png', dpi=300)
            plt.show()
        summarize_metrics(metrics_total, metrics_list, aggregate=False)
        visual_counter += 1

    summarize_metrics(metrics_total, metrics_list, aggregate=True)   
                                           
                                           
## Parse arguments to denoise images and compare results using differnt models, settings, and experiments
def parse_args(argv):
    parser = argparse.ArgumentParser(prog='Inference', add_help=True)
    parser.add_argument('--data_path', type=str, default='/home/ovaheb/projects/def-sdraper/ovaheb/simulated_data/testsets/small_radius')
    parser.add_argument('--result_path', type=str, default='/home/ovaheb/scratch/temp/results')
    parser.add_argument('--model', type=str, nargs='+', action='append', help='Path to the model\'s weight file')
    parser.add_argument('--zsn2n', type=bool, default=False, help='Include ZSN2N results')
    parser.add_argument('--bm3d', type=bool, default=False, help='Include BM3D results')
    parser.add_argument('--filters', type=bool, default=False, help='Include simple filtering methods\' results')
    parser.add_argument('--img_channel', type=int, default=1, help='Number of channels of the image data')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of the patches used for inference')
    #parser.add_argument('--disable_logging', type=bool, default=False, help='Don\'t log results into WandB')
    parser.add_argument('--visualize', type=bool, default=False, help='Show the visualizations')
    parser.add_argument('--structured_noise', type=bool, default=False, help='Add structured noise before inference')
    parser.add_argument('--noise_type', type=str, default='PG', help='P/G/PG/Galsim/None')
    parser.add_argument('--sigma', type=int, default=3, help='Background sigma multiplier for object detection threshold')
    parser.add_argument('--overlap', type=int, default=128, help='Number of overlapping pixels between adjacent windows')
    parser.add_argument('--subtract_bkg', type=bool, default=False, help='Subtract background before inference')
    args = parser.parse_args(argv)
    return args

if __name__=="__main__":
    test(sys.argv[1:])
