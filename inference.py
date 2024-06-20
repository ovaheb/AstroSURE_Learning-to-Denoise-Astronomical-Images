import sys, os
from pathlib import Path
import argparse
import numpy as np
import hashlib
import logging
import random
import time
import datetime
from scipy.stats import entropy
from tqdm import tqdm
import wandb
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.visualization import PercentileInterval, ZScaleInterval, MinMaxInterval
import torch
from denoisers import BaselineDenoiser, UNetDenoiser, DnCNNDenoiser, FilterDenoiser, BM3DDenoiser, ZSN2NDenoiser
from utils import utils_image as util
from utils import utils_logger
#from skimage.restoration import estimate_sigma



def test(argv):
    args = parse_args(argv)
    data_path = args.data_path
    pscale = PercentileInterval(util.PERCENTILE)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    result_path = args.result_path + '/' + args.data_path.split('/')[-1] + '_' + str(date) + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    utils_logger.logger_info(result_path, log_path=result_path + 'log.log')
    logger = logging.getLogger(result_path)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger.info('Run Identifier: %s' %str(date))
    logger.info(args)
    logger.info('All visualizations are done using %.1f%% percentile of images!' %util.PERCENTILE)
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

    for denoiser in denoisers:
        logger.info(denoiser.summarize())
    ### Defining metrics and data path
    metrics_list = ['PSNR', 'SNR', 'SSIM', 'KL Divergence', 'MSE', 'MAE', 'Detection Count', 'False Alarms(%)', 'Reference Count', 'Reference Detected(%)']
    metrics_total['Noisy'] = {metric: [] for metric in metrics_list}
    for denoiser in denoisers:
        metrics_total[denoiser.name] = {metric: [] for metric in metrics_list}
    img_list = [str(file) for file in Path(args.data_path).rglob('*') if (util.is_image_file(str(file)) or util.is_fits_file(str(file)))]

    ################################## Inference ##################################
    rng = np.random.default_rng(int(hashlib.sha256(data_path.encode()).hexdigest(), 16) % 1000) # Generate a random number for each dataset
    height, width, visual_counter = 0, 0, 0
    n_denoisers = len(denoisers)
    for img_name in tqdm(img_list, leave=False, colour='green'):
        ### Reading FITS files
        fits_file = fits.open(img_name)
        if 'JWST' in args.data_path:
            img = np.float32(fits_file['SCI'].data)
            header = fits_file['SCI'].header
            if header['RA_V1'] <= 52.9642:
                continue # Trainng set data
            frame, _, _ = util.read_frame(hf_frame=img, scale_mode=2)
            random_index = random.choice([0, 1])
            other_index = 1 - random_index
            target, _, _ = util.read_frame(hf_frame=frame[random_index:random_index + 1, :, :], scale_mode=2, noise_type='None', header=header)
            source, _, _ = util.read_frame(hf_frame=frame[other_index:other_index + 1, :, :], scale_mode=2, noise_type='None', header=header)
            objs_X, objs_Y, objs_C, objs_D = [], [], [], []
            unsupervised, is_table_hdu = True, False
            skip_detection = True
            nobjs, exptime = 0, 2748
        else:
            primary_hdu_idx = 0 if 'NOBJS' in fits_file[0].header else 1
            header = fits_file[primary_hdu_idx].header
            table_data = fits_file[-1].data
            unsupervised, is_table_hdu = False, True
            skip_detection = True if args.noise_type in ['P', 'G', 'PG'] else False
            nobjs, exptime = header['NOBJS'], header['EXPTIME']
            objs_X, objs_Y, objs_C, objs_D = table_data['Object X'], table_data['Object Y'], table_data['Object Type'], table_data['Object Dimension']
            #objs_A, objs_R, objs_IA = table_data['Object Angle'], table_data['Object Ratio'], table_data['Object Initial Angle']
            #objs_HLR1, objs_HLR2, objs_HLR3 = table_data['Object HLR1'], table_data['Object HLR2'], table_data['Object HLR3']
            target, _, _ = util.read_frame(fits_file, primary_hdu_idx)
            if args.noise_type=='None':
                source, _, _ = util.read_frame(fits_file, primary_hdu_idx + 1, structured_noise=args.structured_noise)
            else:
                source, noise_param1, noise_param2 = util.read_frame(fits_file, primary_hdu_idx, noise_type=args.noise_type, structured_noise=args.structured_noise,
                                                                     rng=rng, header=header, subtract_bkg=args.subtract_bkg)

        target, source = np.transpose(target, (1, 2, 0)), np.transpose(source, (1, 2, 0))
        
        ### Extract patches
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


        if args.combine_patches:
            metrics_baseline, detected_objects_baseline, reference_objects_baseline, used_detections_baseline, assignments_baseline = util.calculate_metrics(target=target, image=source,
                objs_X=objs_X, objs_Y=objs_Y, objs_C=objs_C, objs_D=objs_D, border=args.overlap, sigma_bkg=args.sigma, skip_detection=False, unsupervised=unsupervised)
            for metric_name, metric in zip(metrics_list, metrics_baseline):
                metrics_total['Noisy'][metric_name].append(metric)
        else:
            for top, left in patch_coordinates:
                source_patch = source[top:top + args.patch_size, left:left + args.patch_size, :]
                target_patch = target[top:top + args.patch_size, left:left + args.patch_size, :]
                metrics_baseline, detected_objects_baseline, reference_objects_baseline, used_detections_baseline, assignments_baseline = util.calculate_metrics(target=target_patch,
                    image=source_patch, objs_X=objs_X, objs_Y=objs_Y, objs_C=objs_C, objs_D=objs_D, border=args.overlap, sigma_bkg=args.sigma, skip_detection=True, unsupervised=unsupervised)
                for metric_name, metric in zip(metrics_list, metrics_baseline):
                    metrics_total['Noisy'][metric_name].append(metric)
        ############## Initialize plots for visualization #################
        if args.visualize and visual_counter < util.MAX_NUM_TO_VISUALIZE:
            ### Initialize parameters  for visualization
            logger.info('Source file: %s, Exposure time: %ds, Total objects in FOV: %d'%(img_name, exptime, nobjs))
            if args.combine_patches:
                if 'JWST' in args.data_path:
                    fits_file = fits.HDUList([fits.PrimaryHDU(data=fits_file['SCI'].data[0, :, :]), fits.ImageHDU(data=fits_file['SCI'].data[1, :, :], name='SCI2')])
                elif is_table_hdu:
                    fits_file.insert(len(fits_file) - 1, fits.ImageHDU(data=np.squeeze(source.astype(np.float32))))
                else:
                    fits_file.append(fits.ImageHDU(data=np.squeeze(source.astype(np.float32))))
                source_to_visualize = source
                target_to_visualize = target
            else:
                source_to_visualize = source_patch
                target_to_visualize = target_patch
                    
            bins = 200
            norm = mcolors.LogNorm(vmin=0.1, vmax=1000.0, clip=True)
            num_subplots = n_denoisers + 2
            num_rows = int((num_subplots - 1) / 3) + 1
            num_cols = 3
            if args.noise_type in ['P', 'G', 'PG']:
                logger.info('Gaussian Noise Parameter: %d, Poisson Noise Paramater:%d'%(noise_param1, noise_param2))
                
            ### Plot qualitative result
            fig_qual, axs_qual = plt.subplots(num_rows, num_cols, figsize=(util.PLOT_SIZE*num_cols, util.PLOT_SIZE*num_rows))
            axs_qual = axs_qual.flatten()
            image_obj = axs_qual[0].imshow(pscale(source_to_visualize), interpolation='nearest', cmap='gray', vmin=0, vmax=1)
            axs_qual[0].set_title('Noisy; PSNR=%.2f'%(metrics_total['Noisy']['PSNR'][-1]))
            cmap, _ = image_obj.get_cmap(), plt.colorbar(image_obj, ax=axs_qual[0], fraction=0.046, pad=0.04)
            axs_qual[len(denoisers) + 1].imshow(pscale(target_to_visualize), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
            axs_qual[len(denoisers) + 1].set_title('Ground Truth')
            
            ### Plot detections on images
            fig_obj, axs_obj = plt.subplots(num_rows, num_cols, figsize=(util.PLOT_SIZE*num_cols, util.PLOT_SIZE*num_rows))
            axs_obj = axs_obj.flatten()
            bkg_image = sep.Background(source_to_visualize.squeeze().astype(np.float64))
            bkgsub_image = source_to_visualize.squeeze().astype(np.float64) - bkg_image
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
            bkg_image = sep.Background(target_to_visualize.squeeze().astype(np.float64))
            bkgsub_image = target_to_visualize.squeeze().astype(np.float64) - bkg_image
            axs_obj[len(denoisers) + 1].imshow(pscale(bkgsub_image), interpolation='nearest', cmap=cmap2, vmin=0, vmax=1)
            axs_obj[len(denoisers) + 1].set_title('Ground Truth')
            for reference_x, reference_y, _, reference_d in list(reference_objects_baseline):
                e = Ellipse(xy=(reference_x, reference_y), width=reference_d, height=reference_d, angle=0)
                e.set_facecolor('none')
                e.set_edgecolor('green')
                axs_obj[len(denoisers) + 1].add_artist(e)
            
            ### Plot pixel distributions
            fig_dist, axs_dist = plt.subplots(num_rows, num_cols, figsize=(util.PLOT_SIZE*num_cols, util.PLOT_SIZE*num_rows))
            axs_dist = axs_dist.flatten()
            axs_dist[0].hist(np.ravel(source_to_visualize), bins=bins, histtype='stepfilled')
            axs_dist[0].set_title('Noisy; KL Div.=%.5f'%(metrics_total['Noisy']['KL Divergence'][-1]))
            axs_dist[len(denoisers) + 1].hist(np.ravel(target_to_visualize), bins=bins, histtype='stepfilled')
            axs_dist[len(denoisers) + 1].set_title('Ground Truth')

            ### Plot Error Maps
            num_subplots2 = num_subplots - 1
            num_rows2 = int((num_subplots2 - 1) / 3) + 1
            fig_err, axs_err = plt.subplots(num_rows2, num_cols, figsize=(util.PLOT_SIZE*num_cols, util.PLOT_SIZE*num_rows2))
            axs_err = axs_err.flatten()
            image_obj3 = axs_err[0].imshow(np.abs(source_to_visualize - target_to_visualize), interpolation='nearest', cmap='gray_r', norm=norm)
            axs_err[0].set_title('Noisy; MAE=%.3f'%(metrics_total['Noisy']['MAE'][-1]))
            cmap3, _ = image_obj3.get_cmap(), plt.colorbar(image_obj3, ax=axs_err[0], fraction=0.046, pad=0.04)
            
            ### Plot Residuals
            fig_res, axs_res = plt.subplots(num_rows2, num_cols, figsize=(util.PLOT_SIZE*num_cols, util.PLOT_SIZE*num_rows2))
            axs_res = axs_res.flatten()
            image_obj4 = axs_res[0].imshow(np.abs(target_to_visualize - source_to_visualize), interpolation='nearest', cmap='gray_r', norm=norm)
            axs_err[0].set_title('|GT - Noisy|')
            cmap4, _ = image_obj4.get_cmap(), plt.colorbar(image_obj4, ax=axs_res[0], fraction=0.046, pad=0.04)
            
        
        ################# Inference #########################
        batch_size = 128
        for idx_denoiser in range(len(denoisers)):
            scaled_source, param1, param2 = util.scale(source, denoiser.scaler)
            denoiser = denoisers[idx_denoiser]
            denoised_source = np.zeros_like(scaled_source) if args.combine_patches else []
            denoised_source_count = np.zeros_like(scaled_source)
            start_idx = 0
            while start_idx < len(patch_coordinates):
                end_idx = start_idx + batch_size if start_idx + batch_size<=len(patch_coordinates) else len(patch_coordinates)
                scaled_source_patches = torch.permute(torch.stack([torch.from_numpy(scaled_source[top:top + args.patch_size, left:left + args.patch_size,
                                                        :].astype(np.float32)) for top, left in patch_coordinates[start_idx:end_idx]], dim=0), (0, 3, 1, 2))
                with torch.no_grad():
                    estimated = denoiser.denoise(scaled_source_patches.to(device, non_blocking=True))
                    estimated = util.descale(estimated, denoiser.scaler, param1, param2)
                    if 'JWST' in args.data_path or denoiser.disable_clipping:
                        pass
                    else:
                        pass #estimated = np.clip(estimated, 0.0, 65536.0)
                start_idx += batch_size
                
                if args.combine_patches:
                    for idx_estimated, (top, left) in enumerate(patch_coordinates[start_idx:end_idx]):
                        denoised_source[top:top + args.patch_size, left:left + args.patch_size, :] += estimated[idx_estimated, :, :, :]
                        denoised_source_count[top:top + args.patch_size, left:left + args.patch_size, :] += 1
                else:
                    for idx_estimated in range(estimated.shape[0]):
                        denoised_source.append(estimated[idx_estimated, :, :, :])
            
            if args.combine_patches:
                denoised_source /= denoised_source_count
                metrics, detected_objects_denoiser, reference_objects_denoiser, used_detections_denoiser, assignments_denoiser = util.calculate_metrics(target, denoised_source, objs_X,
                                                                            objs_Y, objs_C, objs_D, args.overlap, args.sigma, skip_detection=skip_detection, unsupervised=unsupervised)
                for metric_name, metric in zip(metrics_list, metrics):
                    metrics_total[denoiser.name][metric_name].append(metric)
                denoised_source_to_visualize = denoised_source
            else:
                for idx_estimated, (top, left) in enumerate(patch_coordinates):
                    target_patch = target[top:top + args.patch_size, left:left + args.patch_size, :]
                    denoised_source_patch = denoised_source[idx_estimated]

                    metrics, detected_objects_denoiser, reference_objects_denoiser, used_detections_denoiser, assignments_denoiser = util.calculate_metrics(target_patch, denoised_source_patch,
                                                                            objs_X, objs_Y, objs_C, objs_D, args.overlap, args.sigma, skip_detection=skip_detection, unsupervised=unsupervised)
                    for metric_name, metric in zip(metrics_list, metrics):
                        metrics_total[denoiser.name][metric_name].append(metric)
                denoised_source_to_visualize = denoised_source_patch

            ########### Visualize denoised images ##########
            if args.visualize and visual_counter < util.MAX_NUM_TO_VISUALIZE:
                idx_image = idx_denoiser + 1
                ### Visualizing images to compare qualitatively
                axs_qual[idx_image].imshow(pscale(denoised_source_to_visualize), interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
                axs_qual[idx_image].set_title('%s; PSNR=%.2f'%(denoiser.name, metrics_total[denoiser.name]['PSNR'][-1]))
                
                ### Show the detected objects in each denoised image
                bkg_image = sep.Background(denoised_source_to_visualize.squeeze().astype(np.float64))
                bkgsub_image = denoised_source_to_visualize.squeeze().astype(np.float64) - bkg_image
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
                axs_dist[idx_image].hist(np.ravel(denoised_source_to_visualize), bins=bins, histtype='stepfilled')
                axs_dist[idx_image].set_title('%s; KL Div.=%.5f'%(denoiser.name, metrics_total[denoiser.name]['KL Divergence'][-1]))
                
                ### Error map
                axs_err[idx_image].imshow(np.abs(denoised_source_to_visualize - target_to_visualize), interpolation='nearest', cmap=cmap3, norm=norm)
                axs_err[idx_image].set_title('%s; MAE=%.3f'%(denoiser.name, metrics_total[denoiser.name]['MAE'][-1]))
                
                ### Residuals
                axs_res[idx_image].imshow(np.abs(denoised_source_to_visualize - source_to_visualize), interpolation='nearest', cmap=cmap4, norm=norm)
                axs_res[idx_image].set_title('|%s - Noisy|'%denoiser.name)
                
                if args.combine_patches:
                    if is_table_hdu:
                        fits_file.insert(len(fits_file) - 1, fits.ImageHDU(data=np.squeeze(denoised_source.astype(np.float32)), name=denoiser.name))
                    elif 'JWST' in args.data_path:
                        fits_file.append(fits.ImageHDU(data=np.squeeze(denoised_source.astype(np.float32)), name=denoiser.name))
                    else:
                        fits_file.append(fits.ImageHDU(data=np.squeeze(denoised_source.astype(np.float32)), name=denoiser.name))
            
        ### Show the visualizations
        if args.visualize and visual_counter < util.MAX_NUM_TO_VISUALIZE:
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
                axs_res[idx_axs].axis('off')
            for idx_axs in range(num_subplots2):
                axs_err[idx_axs].set_xticks([])
                axs_err[idx_axs].set_yticks([])
                axs_res[idx_axs].set_xticks([])
                axs_res[idx_axs].set_yticks([])  
            plt.tight_layout()
            
            file_path = (result_path + img_name.split('/')[-1])[:-5]
            if args.combine_patches:
                fits_file.writeto(file_path + '.fits', overwrite=True)
            fits_file.close()
            fig_qual.savefig(file_path + '_visualizations.png', dpi=300)
            fig_obj.savefig(file_path + '_objects.png', dpi=300)
            fig_dist.savefig(file_path + '_distributions.png', dpi=300)
            fig_err.savefig(file_path + '_error_maps.png', dpi=300)
            fig_res.savefig(file_path + '_residuals.png', dpi=300)
            #plt.show()
        logger.info(util.summarize_metrics(metrics_total, metrics_list, aggregate=False))
        visual_counter += 1

    logger.info(util.summarize_metrics(metrics_total, metrics_list, aggregate=True))
                                           
                                           
## Parse arguments to denoise images and compare results using differnt models, settings, and experiments
def parse_args(argv):
    parser = argparse.ArgumentParser(prog='Inference', add_help=True)
    parser.add_argument('--data_path', type=str, default='/home/ovaheb/projects/def-sdraper/ovaheb/simulated_data/testsets/large_radius')
    parser.add_argument('--result_path', type=str, default='/home/ovaheb/scratch/temp/results')
    parser.add_argument('--model', type=str, nargs='+', action='append', help='Path to the model\'s weight file')
    parser.add_argument('--zsn2n', type=bool, default=False, help='Include ZSN2N results')
    parser.add_argument('--bm3d', type=bool, default=False, help='Include BM3D results')
    parser.add_argument('--filters', type=bool, default=False, help='Include simple filtering methods\' results')
    parser.add_argument('--img_channel', type=int, default=1, help='Number of channels of the image data')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of the patches used for inference')
    parser.add_argument('--visualize', type=bool, default=False, help='Show the visualizations')
    parser.add_argument('--structured_noise', type=bool, default=False, help='Add structured noise before inference')
    parser.add_argument('--noise_type', type=str, default='PG', help='P/G/PG/Galsim/None')
    parser.add_argument('--sigma', type=int, default=3, help='Background sigma multiplier for object detection threshold')
    parser.add_argument('--overlap', type=int, default=128, help='Number of overlapping pixels between adjacent windows')
    parser.add_argument('--subtract_bkg', type=bool, default=False, help='Subtract background before inference')
    parser.add_argument('--combine_patches', type=bool, default=False, help='Combine patches before inference')
    #parser.add_argument('--disable_logging', type=bool, default=False, help='Don\'t log results into WandB')
    args = parser.parse_args(argv)
    return args

if __name__=="__main__":
    test(sys.argv[1:])