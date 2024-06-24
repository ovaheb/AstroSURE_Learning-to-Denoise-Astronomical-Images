import os
import sys
import math
import random
import numpy as np
import torch
import cv2
import sep
from torchvision.utils import make_grid
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
from astropy.visualization import ZScaleInterval, MinMaxInterval, PercentileInterval, AsymmetricPercentileInterval
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from scipy import signal
import galsim
import galsim.roman as roman
import torch.nn as nn
import cr
import torchvision.transforms.functional as tvF
from datetime import date
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import statistics

## Variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams.update({'font.size': 12})
PERCENTILE = 99.9
PLOT_SIZE, MAX_NUM_TO_VISUALIZE = 7, 5

'''
# --------------------------------------------
# Custom Loss Functions
# --------------------------------------------
'''

class CustomLoss(nn.Module):
    def __init__(self, loss_weight=1.0, prior_weight=1.0, loss=nn.L1Loss()):
        super(CustomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.prior_weight = prior_weight
        self.loss = loss
        
    def forward(self, input, target):
        l2_loss = nn.MSELoss()
        return self.loss_weight * self.loss(input, target) + self.prior_weight * torch.mean(torch.abs(input))


'''
# --------------------------------------------
# FITS File Handling
# --------------------------------------------
'''

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
FITS_EXTENSIONS = ['.fits', '.fz', '.fits.fz']
MAX_PIXEL_VALUE = 65536
epsilon = 1e-9

def is_image_hdu(hdu):
    is_image = False
    if isinstance(hdu, fits.CompImageHDU) or isinstance(hdu, fits.ImageHDU):
        is_image = True
    elif isinstance(hdu, fits.PrimaryHDU) and hdu.data is not None:
        if len(hdu.data.shape) > 1:
            is_image = True
    return is_image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_fits_file(filename):
    filename = str(filename).lower()
    return any(filename.endswith(extension) for extension in FITS_EXTENSIONS)




'''
# --------------------------------------------
# image processing process
# --------------------------------------------
'''
def augment(img1, img2):
    augment_idx = random.randint(0, 5) # [0,1,2,3,4,5]
    if augment_idx  < 4:
        return tvF.rotate(img1, 90 * augment_idx), tvF.rotate(img2, 90 * augment_idx)
    elif augment_idx == 4:
        return tvF.hflip(img1), tvF.hflip(img2)
    elif augment_idx == 5:
        return tvF.vflip(img1), tvF.vflip(img2)
    return img1, img2

def remove_nan(image):
    # replace NaN with 0.0 if exist
    if np.sum(np.isnan(image)) > 0:
        image = np.nan_to_num(image, copy=False, nan=0.0)
    return image

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
        reshaped_img = img.reshape(-1, 1)
        sscaler.fit(reshaped_img)
        mu, sigma = sscaler.mean_, sscaler.scale_
        return sscaler.fit_transform(reshaped_img).reshape(img.shape), mu, sigma
    elif scaler == 'arcsinh':
        if isinstance(img, np.ndarray):
            return np.arcsinh(img), None, None
        else:
            return torch.arcsinh(img), None, None
    else:
        raise Exception('Unknown scaler type!')


# Descaler function that scales the denoised data back to the original data range
def descale(img, scaler, param1=None, param2=None):
    if scaler == 'noscale':
        return img
    elif scaler == 'division':
        return img*65536.0
    elif scaler == 'norm':
        return img*(param2-param1) + param1
    elif scaler == 'standard':
        return img*param2 + param1
    elif scaler == 'arcsinh':
        if isinstance(img, np.ndarray):
            return np.sinh(img)
        else:
            return torch.sinh(img)
    else:
        raise Exception('Unknown scaler type!')

'''
# --------------------------------------------
# metric, PSNR, SSIM and PSNRB
# --------------------------------------------
'''


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0, max_pixel=MAX_PIXEL_VALUE):
    # img1 and img2 have range [0, MAX_PIXEL_VALUE]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel / math.sqrt(mse))
    

# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0, max_pixel=MAX_PIXEL_VALUE):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, max_pixel]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    
    if img1.ndim == 2:
        return ssim(img1, img2, max_pixel)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i], max_pixel))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), max_pixel)
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2, max_pixel):
    C1 = (0.01 * max_pixel)**2
    C2 = (0.03 * max_pixel)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def kl_divergence(q, p, is_torch=False):
    # Ensure both probability distributions sum to 1.
    if is_torch:
        p = p / torch.sum(p)
        q = q / torch.sum(q)
    else:
        p = p / np.sum(p)
        q = q / np.sum(q)

    # Avoid division by zero and log of zero.
    p[p <= 0] = epsilon
    q[q <= 0] = epsilon

    # Calculate the KL Divergence.
    if is_torch:
        kl = torch.sum(p * torch.log(p / q))
        return kl
    else:
        kl = np.sum(p * np.log(p / q))
        return kl



'''
# --------------------------------------------
# read image from a frame in FITS file
# --------------------------------------------
'''

def read_frame(fits_img=None, frame_index=None, hf_frame=None, scale_mode=0, noise_type='None', poisson_params=(5,20), gaussian_params=(10,50), structured_noise=False, subtract_bkg=False, rng=None, header=None):
    if frame_index == None:
        frame = hf_frame
    else:
        frame = fits_img[frame_index].data
    frame = np.float32(frame)
    if len(frame.shape) == 2:
        frame = np.expand_dims(frame, axis=0)
    elif len(frame.shape) == 3 and frame.shape[2] <= 5:
        frame = np.transpose(frame, (2, 0, 1))
    frame = remove_nan(frame)
    
    if scale_mode == 0:
        frame = np.clip(frame, 0, MAX_PIXEL_VALUE)
    elif scale_mode == 1:
        frame = np.clip(frame, 0, MAX_PIXEL_VALUE)
        frame[frame>=MAX_PIXEL_VALUE] = 0
    elif scale_mode == 2:
        frame = frame
    else:
        raise ValueError('Wrong scale mode.')
    
    if noise_type != 'None':
        gaussian = rng.uniform(gaussian_params[0], gaussian_params[1])
        poisson = rng.uniform(poisson_params[0], poisson_params[1])
        frame = add_noise(frame, gaussian, poisson, noise_type, rng, header, subtract_bkg)
    
    if structured_noise:
        # TO DO: Implement hot cluster and bad cluster
        nlines = np.random.uniform(5, 21, 4).astype(int)
        # Hot row
        llines = np.random.uniform(300, roman.n_pix+1, nlines[0]).astype(int)
        tlines = np.random.uniform(1, 4, nlines[0]).astype(int)
        for index in range(nlines[0]):
            length = llines[index]
            start = int(np.random.uniform(0, roman.n_pix - length))
            for line_index in range(tlines[index]):
                frame[int(np.random.uniform(0, roman.n_pix - 3)), start:start+length] = np.ones((1, length)).T*500
        # Dead row
        llines = np.random.uniform(300, roman.n_pix+1, nlines[0]).astype(int)
        tlines = np.random.uniform(1, 4, nlines[0]).astype(int)
        for index in range(nlines[0]):
            length = llines[index]
            start = int(np.random.uniform(0, roman.n_pix - length))
            for line_index in range(tlines[index]):
                frame[int(np.random.uniform(0, roman.n_pix - 3)), start:start+length] = np.zeros((1, length)).T
            
        # Hot column
        llines = np.random.uniform(300, roman.n_pix+1, nlines[2]).astype(int)
        tlines = np.random.uniform(1, 4, nlines[2]).astype(int)
        for index in range(nlines[2]):
            length = llines[index]
            start = int(np.random.uniform(0, roman.n_pix - length))
            for line_index in range(tlines[index]):
                frame[start:start+length, int(np.random.uniform(0, roman.n_pix - 3))] = np.ones((length, 1))*500
            
        # Dead column
        llines = np.random.uniform(300, roman.n_pix+1, nlines[3]).astype(int)
        tlines = np.random.uniform(1, 4, nlines[3]).astype(int)
        for index in range(nlines[3]):
            length = llines[index]
            start = int(np.random.uniform(0, roman.n_pix - length))
            for line_index in range(tlines[index]):
                frame[start:start+length, int(np.random.uniform(0, roman.n_pix - 3))] = np.zeros((length, 1))
            
        
        npixels = np.random.uniform(0.0002*roman.n_pix**2, 0.0005*roman.n_pix**2, 2).astype(int)
        total_pixels = roman.n_pix * roman.n_pix
        npixels_hot = random.sample(range(total_pixels), npixels[0])
        npixels_dead = random.sample(range(total_pixels), npixels[1])
        # Hot pixels
        for index in npixels_hot:
            row = index // roman.n_pix
            col = index % roman.n_pix
            frame[row, col] = 500
        
        # Dead pixels
        for index in npixels_dead:
            row = index // roman.n_pix
            col = index % roman.n_pix
            frame[row, col] = 0
        
        # Cosmic rays
        cr_flux = 8  # events/cm^2/s
        wfi_area = 16.8  # cm^2
        t_exp = 3  # s
        frame = np.expand_dims(cr.simulate_crs(np.squeeze(frame), cr_flux, wfi_area, t_exp), axis=2)
        
    if noise_type=='PG':
        return frame, gaussian, poisson
    elif noise_type=='G':
        return frame, gaussian, 0
    elif noise_type=='P':
        return frame, 0, poisson
    elif noise_type=='Galsim':
        return frame, 0, 0
    else:
        return frame, 0, 0



'''
# --------------------------------------------
# Add noise with multiple modalities
# --------------------------------------------
'''
def add_noise(img, gaussian, poisson, noise_type, rng, header=None, subtract_bkg=False, idx_noisy=0):
    '''
    Add noise to the input image.
    '''
    if noise_type=='PG':
        noise_img = rng.poisson(lam=img/poisson, size=img.shape) * poisson + rng.normal(loc=0, scale=gaussian, size=img.shape)
    elif noise_type=='G':
        noise_img = img + rng.normal(loc=0, scale=gaussian, size=img.shape)
    elif noise_type=='P':
        noise_img = rng.poisson(lam=img/poisson, size=img.shape) * poisson
    elif noise_type=='Galsim':
        noise_img = add_noise_galsim(img, rng, header, subtract_bkg, idx_noisy)
    else:
        noise_img = img
    return noise_img

def add_noise_galsim(img, rng, header, subtract_bkg=False, idx_noisy=0):
    random_seed = int(rng.random()*10**8)
    seed = galsim.BaseDeviate(random_seed).raw()
    image_rng = galsim.UniformDeviate(7*seed*(idx_noisy + 1) + header['NOBJS'])
    noisy_realization = galsim.Image(np.squeeze(img), copy=True)
    poisson_noise = galsim.PoissonNoise(image_rng)
    exptime = header['EXPTIME']
    targ_pos = galsim.CelestialCoord(ra=header['RA']*galsim.degrees, dec=header['DEC']*galsim.degrees)
    wcs = roman.getWCS(world_pos=targ_pos, SCAs=7, date=date.fromisoformat(header['DATE'][:10]))[7]
    sky_image = galsim.ImageF(4088, 4088, wcs=wcs)
    SCA_cent_pos = wcs.toWorld(sky_image.true_center)
    sky_level = roman.getSkyLevel(roman.getBandpasses(AB_zeropoint=True)['Y106'], world_pos=SCA_cent_pos)
    sky_level *= (1.0 + roman.stray_light_fraction)
    wcs.makeSkyImage(sky_image, sky_level)
    sky_image += roman.thermal_backgrounds['Y106']*exptime
    sky_image.addNoise(poisson_noise)
    noisy_realization += sky_image
    roman.addReciprocityFailure(noisy_realization)
    dark_current = roman.dark_current*exptime
    dark_noise = galsim.DeviateNoise(galsim.PoissonDeviate(image_rng, dark_current))
    noisy_realization.addNoise(dark_noise)
    roman.applyNonlinearity(noisy_realization)
    roman.applyIPC(noisy_realization)
    read_noise = galsim.GaussianNoise(image_rng, sigma=roman.read_noise)
    noisy_realization.addNoise(read_noise)
    if subtract_bkg:
        noisy_realization -= sky_image
    ### Gain and quantization
    noisy_realization /= roman.gain
    noisy_realization.quantize()
    
    return np.expand_dims(noisy_realization.array, axis=0)

'''
# --------------------------------------------
# Functions to measure denoising quality
# --------------------------------------------
'''

def plot_detected_objects(img_E_data, img_E_objects, img_H_data, img_H_objects, plot_size, gt_objs):
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(plot_size*2,plot_size), ncols=2)
    m, s = np.mean(img_E_data), np.std(img_E_data)
    im = ax[0].imshow(img_E_data, interpolation='nearest', vmin=m-s, vmax=m+s, cmap='gray')
    ax[0].set_title("Detected objects from denoised : {} object(:s)".format(len(img_E_objects)))
    # plot an ellipse for each object
    for i in range(len(img_E_objects)):
        e = Ellipse(xy=(img_E_objects['x'][i], img_E_objects['y'][i]), width=6*img_E_objects['a'][i],
                    height=6*img_E_objects['b'][i], angle=img_E_objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax[0].add_artist(e)
    ax[0].set_xticks([])
    ax[0].set_yticks([])           
    m, s = np.mean(img_H_data), np.std(img_H_data)
    im = ax[1].imshow(img_H_data, interpolation='nearest', vmin=m-s, vmax=m+s, cmap='gray')
    ax[1].set_title("Objects in ground truth : {} object(:s)".format(gt_objs if gt_objs is not None else len(img_H_objects)))
    # plot an ellipse for each object
    if gt_objs == None:
        for i in range(len(img_H_objects)):
            e = Ellipse(xy=(img_H_objects['x'][i], img_H_objects['y'][i]), width=6*img_H_objects['a'][i],
                        height=6*img_H_objects['b'][i], angle=img_H_objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax[1].add_artist(e)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.subplots_adjust(wspace=0)
    plt.show()
    
def get_error_map(denoised, clean, exptime=140, mode='MAE'):
    if not denoised.shape == clean.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = denoised.shape[:2]
    if mode == 'MAE':
        error = np.abs(denoised - clean)
    elif mode == 'MSE':
        error = (denoised - clean)**2
    
    clean = clean.flatten()
    clean[clean == 0] = epsilon
    clean = clean.reshape(h, w)
    

'''
# --------------------------------------------
# Estimating variance of noise (Gaussian)
# --------------------------------------------
# Immerkær, John. “Fast Noise Variance Estimation.” Comput. Vis. Image Underst. 64 (1996): 300-302.
# https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
# --------------------------------------------
'''

def estimate_noise_level(I):
    H, W = I.shape[:2]
    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(signal.convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5*math.pi) / (6*(W - 2)*(H - 2))
    return np.sqrt(sigma)

'''
# --------------------------------------------
# Anscombe and Generalized Anscome Transform
# --------------------------------------------
'''

def anscombe(x):
    '''
    Compute the anscombe variance stabilizing transform.

      the input   x   is noisy Poisson-distributed data
      the output  fx  has variance approximately equal to 1.

    Reference: Anscombe, F. J. (1948), "The transformation of Poisson,
    binomial and negative-binomial data", Biometrika 35 (3-4): 246-254
    '''
    x = np.maximum(x, -3.0/8.0)
    return 2.0*np.sqrt(x + 3.0/8.0)

def inverse_anscombe(z):
    '''
    Compute the inverse transform using an approximation of the exact
    unbiased inverse.

    Reference: Makitalo, M., & Foi, A. (2011). A closed-form
    approximation of the exact unbiased inverse of the Anscombe
    variance-stabilizing transformation. Image Processing.
    '''
    z =  np.maximum(z, 0)
    return (1.0/4.0 * np.power(z, 2) + 1.0/4.0 * np.sqrt(3.0/2.0) * np.power(z, -1.0) -
            11.0/8.0 * np.power(z, -2.0) + 5.0/8.0 * np.sqrt(3.0/2.0) * np.power(z, -3.0) - 1.0 / 8.0)

def generalized_anscombe(x, mu, sigma, gain=1.0):
    '''
    Compute the generalized anscombe variance stabilizing transform,
    which assumes that the data provided to it is a mixture of poisson
    and gaussian noise.

    The input signal  z  is assumed to follow the Poisson-Gaussian noise model

        x = gain * p + n

    where gain is the camera gain and mu and sigma are the read noise
    mean and standard deviation.

    We assume that x contains only positive values.  Values that are
    less than or equal to 0 are ignored by the transform.

    Note, this transform will show some bias for counts less than
    about 20.
    '''
    y = gain*x + (gain**2)*3.0/8.0 + sigma**2 - gain*mu

    # Clamp to zero before taking the square root.
    return (2.0/gain)*np.sqrt(np.maximum(y, 0.0))

def inverse_generalized_anscombe(x, mu, sigma, gain=1.0):
    '''
    Applies the closed-form approximation of the exact unbiased
    inverse of Generalized Anscombe variance-stabilizing
    transformation.

    The input signal x is transform back into a Poisson random variable
    based on the assumption that the original signal from which it was
    derived follows the Poisson-Gaussian noise model:

        x = gain * p + n

    where gain is the camera gain and mu and sigma are the read noise
    mean and standard deviation.

    Roference: M. Makitalo and A. Foi, "Optimal inversion of the
    generalized Anscombe transformation for Poisson-Gaussian noise",
    IEEE Trans. Image Process., doi:10.1109/TIP.2012.2202675

    '''
    test = np.maximum(x, 1.0)
    exact_inverse = (np.power(test/2.0, 2.0) + 1.0/4.0 * np.sqrt(3.0/2.0) * np.power(test, -1.0) - 11.0/8.0 * np.power(test, -2.0) + 
                     5.0/8.0 * np.sqrt(3.0/2.0) * np.power(test, -3.0) - 1.0/8.0 - np.power(sigma, 2))
    exact_inverse = np.maximum(0.0, exact_inverse)
    exact_inverse *= gain
    exact_inverse += mu
    exact_inverse[np.where(exact_inverse != exact_inverse)] = 0.0
    return exact_inverse

def uMSE(image, target):
    #source_b, source_c = extract_neighbors(source)
    return np.mean((image - target) ** 2) #- np.mean((source_b - source_c) ** 2) / 2

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
def calculate_metrics(target, image, objs_X, objs_Y, objs_C, objs_D, border=128, sigma_bkg=3, skip_detection=False, unsupervised=False):
    pscale = PercentileInterval(PERCENTILE)
    if unsupervised:
        mse = uMSE(image, target)
    else:
        mse = np.mean((target - image)**2)
    mae = np.mean(np.abs(target - image))
    max_pixel = np.max(target) if unsupervised else 65536.0
    psnr = 20*math.log10(max_pixel / math.sqrt(mse)) if mse >= 0 else None
    snr = 10*np.log10(np.sum(image**2) / np.sum((image - target)**2))
    ssim = calculate_ssim(target.squeeze(), image.squeeze(), max_pixel=max_pixel)
    kl = kl_divergence(target.ravel(), image.ravel())
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

# Function to print the metrics
def summarize_metrics(model_results, metrics_list, aggregate=False):
    model_names = list(model_results.keys())
    headers = ['Model'] + metrics_list
    if aggregate:
        table_data = [[model] + [statistics.mean(metrics[model_name]) for model_name in metrics_list] for model, metrics in model_results.items()]
        return '\n' + tabulate(table_data, headers, tablefmt='fancy_grid')
    else:
        table_data = [[model] + [metrics[metric][-1] for metric in metrics_list] for model, metrics in model_results.items()]
        return '\n' + tabulate(table_data, headers)




if __name__ == '__main__':
    pass