import os
import sys
import math
import random
import numpy as np
import torch
import cv2
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams.update({'font.size': 12})

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

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def imshow(x, title=None, cbar=False, figsize=None, font_scale=1, hide_ticks=True):
    font_size = 18 // font_scale
    plt.rcParams.update({'font.size': font_size})

    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    cmap = plt.gca().get_images()[0].get_cmap()
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar(fraction=0.046, pad=0.04)
    if hide_ticks:
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return cmap


'''
# --------------------------------------------
# get image pathes
# --------------------------------------------
'''


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) or is_fits_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has nothing valid image file'.format(path)
    return images


'''
# --------------------------------------------
# split large images into small images 
# --------------------------------------------
'''


def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))
        w1.append(w-p_size)
        h1.append(h-p_size)
        for i in w1:
            for j in h1:
                patches.append(img[i:i+p_size, j:j+p_size,:])
    else:
        patches.append(img)
    return patches

def patches_from_image_coordinate(img, index, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))
        w1.append(w-p_size)
        h1.append(h-p_size)
        for i in w1:
            for j in h1:
                patches.append((i,i+p_size, j,j+p_size))
    else:
        patches.append(img)
    return patches[index]


def imssave(imgs, img_path):
    """
    imgs: list, N images of size WxHxC
    """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        new_path = os.path.join(os.path.dirname(img_path), img_name+str('_{:04d}'.format(i))+'.png')
        cv2.imwrite(new_path, img)



'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(uint)
# numpy(single) <--->  tensor
# numpy(uint)   <--->  tensor
# --------------------------------------------
'''


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------


def uint2single(img):

    return np.float32(img/MAX_PIXEL_VALUE)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*MAX_PIXEL_VALUE).round())


def uint162single(img):

    return np.float32(img/65535.)


def single2uint16(img):

    return np.uint16((img.clip(0, 1)*65535.).round())


# --------------------------------------------
# numpy(uint) (HxWxC or HxW) <--->  tensor
# --------------------------------------------


# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(MAX_PIXEL_VALUE).unsqueeze(0)


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(MAX_PIXEL_VALUE)


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


# convert torch tensor to single
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

# convert torch tensor to single
def tensor2single3(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float().unsqueeze(0)


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()

"""
# from skimage.io import imread, imsave
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array of BGR channel order
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # squeeze first, then clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.uint8() WILL NOT round by default.
    return img_np.astype(out_type)
"""

'''
# --------------------------------------------
# Augmentation, flipe and/or rotate
# --------------------------------------------
# The following two are enough.
# (1) augmet_img: numpy image of WxHxC or WxH
# (2) augment_img_tensor4: tensor image 1xCxWxH
# --------------------------------------------
'''


def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def augment_img_tensor4(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


def augment_img_tensor(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img_np3(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.transpose(1, 0, 2)
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 4:
        return img[:, ::-1, :]
    elif mode == 5:
        img = img[:, ::-1, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 6:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        return img
    elif mode == 7:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img


def augment_imgs(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def augment(img1, img2):
    augment_idx = random.randint(0, 5) # [0,1,2,3,4,5]
    if augment_idx  < 4:
        return tvF.rotate(img1, 90 * augment_idx), tvF.rotate(img2, 90 * augment_idx)
    elif augment_idx == 4:
        return tvF.hflip(img1), tvF.hflip(img2)
    elif augment_idx == 5:
        return tvF.vflip(img1), tvF.vflip(img2)
    return img1, img2

'''
# --------------------------------------------
# modcrop and shave
# --------------------------------------------
'''


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def shave(img_in, border=0):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img


'''
# --------------------------------------------
# image processing process
# --------------------------------------------
'''
def remove_nan(image):
    # replace NaN with 0.0 if exist
    if np.sum(np.isnan(image)) > 0:
        image = np.nan_to_num(image, copy=False, nan=0.0)
    return image

def scale(img, scaler):
    if scaler == 'noscale':
        return img
    elif scaler == 'division':
        return img/65536.0
    elif scaler == 'norm':
        mmscale = MinMaxInterval()
        return mmscale(img)
    elif scaler == 'standard':
        reshaped_img = img.reshape(-1, 1)
        sscaler = StandardScaler()
        #sscaler.fit(reshaped_img)
        return sscaler.fit_transform(reshaped_img).reshape(img.shape)
    elif scaler == 'arcsinh':
        return np.arcsinh(img)


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
        return torch.sinh(img)

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
    
    if len(frame.shape) == 2:
        frame = np.expand_dims(np.float32(frame), axis=0)
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
    
    #return np.clip(error / clean, 0, 100)
    #return error / ((roman.read_noise + roman.dark_current*exptime + roman.thermal_backgrounds['Y106']*exptime) / roman.gain)
    
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

if __name__ == '__main__':
    pass