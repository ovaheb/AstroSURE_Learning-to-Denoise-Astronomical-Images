import os
import subprocess
import tempfile
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
import cv2
import sep
from torchvision.utils import make_grid
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
from astropy.visualization import ZScaleInterval, MinMaxInterval, PercentileInterval, AsymmetricPercentileInterval
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import rcParams
from shapely.geometry import Point, Polygon
from scipy import signal
from scipy.ndimage import median_filter
from scipy.interpolate import griddata
import skvideo.measure
import galsim
import galsim.roman as roman
import torch.nn as nn
import cr
import torchvision.transforms.functional as tvF
from datetime import date
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import statistics
import re
import warnings
from astropy.wcs import FITSFixedWarning

sep.set_extract_pixstack(30000000)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="RADECSYS= 'FK5 ' / Coordinate system for equinox (FK4/FK5/GAPPT)", category=FITSFixedWarning, append=True)
warnings.filterwarnings('ignore', message="Invalid parameter values: MJD-OBS and DATE-OBS are inconsistent", category=FITSFixedWarning, append=True)

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
    if np.sum(np.isnan(image)) > 0:
        image = np.nan_to_num(image, copy=False, nan=0.0)
    return image

def remove_nan_CCD(image, method='zero'):
    image = np.where(image < 100, np.nan, image)
    if np.sum(np.isnan(image)) > 0:
        
        if method == 'zero':
            image = np.nan_to_num(image, copy=False, nan=0.0)
        elif method == 'median':
            nan_mask = np.isnan(image)
            median_filtered = median_filter(image, size=7, mode='mirror')
            image[nan_mask] = median_filtered[nan_mask]
        else:
            nan_mask = np.isnan(image)
            interpolated_values = griddata(np.array(np.nonzero(~nan_mask)).T, image[~nan_mask], np.array(np.nonzero(nan_mask)).T, method=method)
            image[nan_mask] = interpolated_values
    return image

def normalize_CCD_range(image, bias=0):
    minimum = np.min(image)
    maximum = np.max(image)
    normalized_image = (image - minimum) / (maximum - minimum)
    scaled_image = normalized_image * 65536.0 + bias
    return scaled_image

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

def read_frame(fits_img=None, frame_index=None, hf_frame=None, scale_mode=0, noise_type='None', poisson_params=(5,20), gaussian_params=(10,50), structured_noise=False, subtract_bkg=False, rng=None, header=None, nan_removal='zero'):
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
    noisy_realization /= roman.gain
    noisy_realization.quantize()
    return np.expand_dims(noisy_realization.array, axis=0)

    

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

def convert_to_degrees(ra_hms, dec_dms):
    coord = SkyCoord(ra=ra_hms, dec=dec_dms, unit=(u.hourangle, u.deg))
    return coord.ra.deg, coord.dec.deg

def is_inside_polygon(polygon, ra, dec):
    point = Point(ra, dec)
    return polygon.contains(point)

def filter_objects(catalog, header, aorb, wcs):
    if aorb == 'A':
        x1, x2, y1, y2 = 32, 1055, 3, 4610
    elif aorb == 'B':
        x1, x2, y1, y2 = 1056, 2080, 3, 4610
    ra1, dec1 = wcs.wcs_pix2world(x1, y1, 0)
    ra2, dec2 = wcs.wcs_pix2world(x2, y1, 0)
    ra3, dec3 = wcs.wcs_pix2world(x2, y2, 0)
    ra4, dec4 = wcs.wcs_pix2world(x1, y2, 0)
    vertices = [(ra1, dec1), (ra2, dec2), (ra3, dec3), (ra4, dec4)]
    polygon = Polygon(vertices)
    start_idx = np.searchsorted(catalog['RA'], min(ra1, ra2, ra3, ra4), side='left')
    end_idx = np.searchsorted(catalog['RA'], max(ra1, ra2, ra3, ra4), side='right')
    subset_catalog = catalog[start_idx:end_idx]
    inside_mask = np.array([is_inside_polygon(polygon, ra, dec) for ra, dec in zip(subset_catalog['RA'], subset_catalog['DEC'])])
    filtered_catalog = subset_catalog[inside_mask]
    coords = SkyCoord(ra=filtered_catalog['RA'], dec=filtered_catalog['DEC'], unit='deg')
    x, y = wcs.world_to_pixel(coords)
    filtered_catalog['X'] = x - x1
    filtered_catalog['Y'] = y - y1
    filtered_catalog['MAJOR'] = filtered_catalog['R_MAG_AUTO']
    filtered_catalog['MINOR'] = filtered_catalog['R_MAG_AUTO']
    filtered_catalog['ANGLE'] = filtered_catalog['R_MAG_AUTO']
    return filtered_catalog


def uMSE(image, target):
    image_d = image[1::2, 1::2].flatten()
    target_a, target_b, target_c = target[::2, ::2].flatten(), target[1::2, ::2].flatten(), target[::2, 1::2].flatten()
    return np.mean((image_d - target_a) ** 2) - np.mean((target_b - target_c) ** 2) / 2

# Function to compute the different metrics
def calculate_metrics(target, image, header, catalog, aorb=None, border=128, sigma_bkg=3, unsupervised=False, elliptical=False, source=None):
    bkg_image = sep.Background(image.squeeze().astype(np.float64))
    bkgsub_image = image.squeeze().astype(np.float64) - bkg_image
    try:
        objects = sep.extract(bkgsub_image, sigma_bkg, err=sep.Background(source.squeeze().astype(np.float64)).rms())#bkg_image.rms())
    except:
        objects = np.empty(0, dtype=[('x', float), ('y', float), ('a', float), ('b', float), ('theta', float)])

    objects = objects[np.maximum(objects['a'], objects['b']) < 100]
    devnull = open(os.devnull, 'w')
    detection_results = tempfile.NamedTemporaryFile(suffix='.fits')
    catalog_results = tempfile.NamedTemporaryFile(suffix='.fits')
    matching_results = tempfile.NamedTemporaryFile(suffix='.fits')
    if unsupervised:
        pixel_scale = (header['PIXSCAL1'] + header['PIXSCAL2']) / 2
        wcs = WCS(header)
        
        catalog_table = filter_objects(catalog, header, aorb, wcs)
        catalog_table.write(catalog_results.name, overwrite=True) #('X', 'Y', 'RA', 'DEC', 'MAJOR', 'MINOR', 'ANGLE', 'R_MAG_AUTO')
        
        if aorb == 'A':
            ra, dec = wcs.wcs_pix2world(objects['x'] + 32, objects['y'] + 3, 0)
        elif aorb == 'B':
            ra, dec = wcs.wcs_pix2world(objects['x'] + 1056, objects['y'] + 3, 0)
        else:
            raise ValueError('Unknown amplifier!')
        detection_table = Table([objects['x'], objects['y'], ra, dec, objects['a'], objects['b'], np.degrees(objects['theta'])], names=('X', 'Y', 'RA', 'DEC', 'MAJOR', 'MINOR', 'ANGLE'))
        detection_table.write(detection_results.name, overwrite=True)
        
        process = subprocess.Popen(['/home/ovaheb/code/topcat/stilts', 'tmatch2', 'in1=' + detection_results.name, 'in2=' + catalog_results.name, 'matcher=sky', 'params=1', 'values1=RA DEC', 
                                    'values2=RA DEC', 'find=best', 'join=1or2', 'progress=none', 'omode=out', 'out=' + matching_results.name, 'suffix1=_DET', 'suffix2=_CAT'], stderr=devnull)
        
    else:
        wcs = WCS(naxis=2)
        pixel_scale = roman.pixel_scale #arcsec/pixel
        wcs.wcs.crpix = [header['NAXIS1'] / 2, header['NAXIS2'] / 2]
        wcs.wcs.crval = [header['RA'], header['DEC']]
        wcs.wcs.cdelt = np.array([-pixel_scale / 3600, pixel_scale / 3600])
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]    
        
        ra, dec = wcs.wcs_pix2world(objects['x'], objects['y'], 0)
        detection_table = Table([objects['x'], objects['y'], ra, dec, objects['a']*pixel_scale, objects['b']*pixel_scale, np.degrees(objects['theta'])], names=('X', 'Y', 'RA', 'DEC', 'MAJOR', 'MINOR', 'ANGLE'))
        detection_table.write(detection_results.name, overwrite=True)
        
        ra2, dec2 = wcs.wcs_pix2world(catalog['Object X'], catalog['Object Y'], 0)
        catalog_table = Table([catalog['Object X'], catalog['Object Y'], ra2, dec2, catalog['Object Dimension']*pixel_scale, catalog['Object Dimension']*catalog['Object Ratio']*pixel_scale,
                             np.degrees(catalog['Object Angle'] + catalog['Object Initial Angle']), catalog['Object Type']], names=('X', 'Y', 'RA', 'DEC', 'MAJOR', 'MINOR', 'ANGLE', 'Type'))
        catalog_table.write(catalog_results.name, overwrite=True)
    
        if elliptical:
            process = subprocess.Popen(['/home/ovaheb/code/topcat/stilts', 'tmatch2', 'in1=' + detection_results.name, 'in2=' + catalog_results.name, 'matcher=skyellipse', 'params=' +
                                        str(np.mean(catalog['Object Dimension'])*pixel_scale), 'values1=RA DEC MAJOR MINOR ANGLE', 'values2=RA DEC MAJOR MINOR ANGLE', 'find=best',
                                        'join=1or2', 'progress=none', 'omode=out', 'out=' + matching_results.name, 'suffix1=_DET', 'suffix2=_CAT'], stderr=devnull)
        else:
            process = subprocess.Popen(['/home/ovaheb/code/topcat/stilts', 'tmatch2', 'in1=' + detection_results.name, 'in2=' + catalog_results.name, 'matcher=2d', 'params=9', 'values1=' + 'X Y', 
                            'values2=' + 'X Y', 'find=best', 'join=1or2', 'progress=none', 'omode=out', 'out=' + matching_results.name, 'suffix1=_DET', 'suffix2=_CAT'], stderr=devnull)
        
    if unsupervised:
        mse = uMSE(image.squeeze(), target.squeeze())
    else:
        mse = np.mean((target - image)**2)
    mae = np.mean(np.abs(target - image))
    max_pixel = np.max(target) if unsupervised else 65536.0
    psnr = 20*math.log10(max_pixel / math.sqrt(mse)) if mse >= 0 else None
    snr = 10*np.log10(np.sum(image**2) / np.sum((image - target)**2))
    ssim = calculate_ssim(target.squeeze(), image.squeeze(), max_pixel=max_pixel)
    kl = kl_divergence(target.ravel(), image.ravel())
    niqe_metric = skvideo.measure.niqe(image)
    _ = process.communicate()
    with fits.open(matching_results.name) as matching_results_fits:
        results = matching_results_fits[1].data
        results = pd.DataFrame(results)
        if not unsupervised:
            results = results.applymap(lambda x: np.nan if isinstance(x, (int, float)) and x < -100000 else x)
            results['MAJOR_DET'], results['MINOR_DET'] = results['MAJOR_DET']/pixel_scale, results['MINOR_DET']/pixel_scale
            results['MAJOR_CAT'], results['MINOR_CAT'] = results['MAJOR_CAT']/pixel_scale, results['MINOR_CAT']/pixel_scale
            
    detected_objs = results['RA_DET'].notnull().sum()
    all_objs = results['RA_CAT'].notnull().sum()
    correct_detections = results['Separation'].notnull().sum()
    #missed = results['RA_DET'].isnull().sum()
    false_alarm_rate = 100.0*results['RA_CAT'].isnull().sum()/detected_objs if detected_objs > 0 else 100
    detection_results.close()
    catalog_results.close()
    matching_results.close()
    return [psnr, snr, ssim, kl, mse, mae, niqe_metric, detected_objs, false_alarm_rate, all_objs, 100.0*correct_detections/all_objs], results

# Function to print the metrics
def summarize_metrics(model_results, metrics_list, aggregate=False):
    headers = ['Model'] + metrics_list
    if aggregate:
        table_data = [[model] + [np.mean(metrics[metric_name]) for metric_name in metrics_list] for model, metrics in model_results.items()]
        return '\n' + tabulate(table_data, headers, tablefmt='fancy_grid')
    else:
        table_data = [[model] + [metrics[metric_name][-1] for metric_name in metrics_list] for model, metrics in model_results.items()]
        return '\n' + tabulate(table_data, headers)



if __name__ == '__main__':
    pass