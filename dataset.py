import os
import torch
from skimage import io
from skimage.transform import resize
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
import sys
from utils import utils_image as util
from astropy.io import fits
from astropy.visualization import PercentileInterval, ZScaleInterval, MinMaxInterval, BaseInterval
import time
from sklearn.preprocessing import StandardScaler
import torchvision.transforms.functional as tvF
import math
import fitsio
import h5py
import json
import galsim
import galsim.roman as roman

MAX_PIXEL_VALUE = 65536.0
def extract_image_size(path):
    if util.is_fits_file(path):
        _ = fits.open(path)
        axis1, axis2 = _[1].header['NAXIS1'], _[1].header['NAXIS2']
        _.close()
    
    elif util.is_image_file(path):
        axis1, axis2 =  256, 256
    return (axis1, axis2)

    
################## Training using HDF5 file ####################
class TrainingDataset(Dataset):
    def __init__(self, hf, data_path, image_list, patch_size, supervised, scaler, img_channel, noise_type='PG', poisson_settings=20, gaussian_settings=50, exptime_division=False, natural=False, subtract_bkg=False):
        '''
        Dataset class for training with ground truth frames and added noise.
        '''
        self.rng = np.random.default_rng(1024)
        self.data_path = data_path
        self.hf = hf
        self.image_list = image_list
        self.patch_size = patch_size
        self.supervised = supervised
        self.scaler = scaler
        self.img_channel = img_channel
        self.natural = natural
        self.exptime_division = exptime_division
        self.subtract_bkg = subtract_bkg
        self.image_size = (4088, 4088) if 'CFHT' not in data_path else (4581, 1024)
        self.num_of_rows, self.num_of_cols = math.ceil(self.image_size[0]/self.patch_size), math.ceil(self.image_size[1]/self.patch_size)
        self.patch_per_image = self.num_of_rows*self.num_of_cols
        self.gaussian_noise_level, self.poisson_noise_level = gaussian_settings, poisson_settings
        self.gaussian_sample, self.poisson_sample = 0, 0
        self.noise_type = noise_type
        self.batch_counter, self.image_counter = 0, 0
        self.clean_image, self.noisy_image, self.noisy_image2 = None, None, None
        self.read_image()
        
    def __len__(self):
        return self.patch_per_image*len(self.image_list)
            
    def read_image(self):
        '''
        Read and preprocess images from the dataset.
        '''
        self.clean_image = self.hf[self.image_list[self.image_counter]]
        header = json.loads(self.clean_image.attrs['Header'])
        scale_mode = 2 if 'JWST' in self.data_path else 0
        self.clean_image, _, _ = util.read_frame(hf_frame=self.clean_image, scale_mode=scale_mode, noise_type='None', header=header)
        if self.clean_image.shape[0] != 1:
            random_index = random.choice([0, 1])
            self.clean_image = self.clean_image[random_index:random_index+1, :, :]

        if 'JWST' in self.data_path:
            other_index = 1 - random_index
            self.noisy_image = util.scale(self.clean_image, self.scaler)[0]
            self.noisy_image2, _, _ = util.read_frame(hf_frame=self.hf[self.image_list[self.image_counter]][other_index:other_index+1, :, :], scale_mode=scale_mode, noise_type='None', header=header)
            self.noisy_image2 = util.scale(self.noisy_image2, self.scaler)[0]
            self.gaussian_sample = header['VAR_RNOISE']
            self.poisson_sample = header['VAR_POISSON']
        elif 'CFHT' in self.data_path:
            self.gaussian_sample = header['gaussian']
            self.poisson_sample = header['poisson']
            self.noisy_image = util.scale(self.clean_image, self.scaler)[0]
            self.noisy_image2 = util.scale(self.clean_image, self.scaler)[0]
        else:
            self.gaussian_sample = self.rng.uniform(0, self.gaussian_noise_level) if self.gaussian_noise_level != None else 0
            self.poisson_sample = self.rng.uniform(0, self.poisson_noise_level) if self.poisson_noise_level != None else 0
            self.noisy_image = util.scale(util.add_noise(self.clean_image, self.gaussian_sample, self.poisson_sample, self.noise_type, self.rng, header, self.subtract_bkg, 1), self.scaler)[0]
            self.noisy_image2 = util.scale(util.add_noise(self.clean_image, self.gaussian_sample, self.poisson_sample, self.noise_type, self.rng, header, self.subtract_bkg, 2), self.scaler)[0]
        self.clean_image = util.scale(self.clean_image, self.scaler)[0]
        self.image_counter += 1
        if self.image_counter==len(self.image_list):
            random.shuffle(self.image_list)
            self.image_counter = 0
        
    def __getitem__(self, idx):
        '''
        Get a sample from the dataset.
        '''
        idx = idx % self.patch_per_image
        row = idx//(self.num_of_cols)
        col = idx%(self.num_of_cols)
        if self.supervised=='N2C':
            img1 = self.clean_image
        else:
            img1 = self.noisy_image
        img2 = self.noisy_image2
        if self.exptime_division:
            img1 /= self.current_exptime
            img2 /= self.current_exptime
        top, left = row*self.patch_size, col*self.patch_size
        top = (self.image_size[0] - self.patch_size) if (top + self.patch_size) >= self.image_size[0] else top
        left = (self.image_size[1] - self.patch_size) if (left + self.patch_size) >= self.image_size[1] else left
        img1 = img1[:, top:top + self.patch_size, left:left + self.patch_size]
        img2 = img2[:, top:top + self.patch_size, left:left + self.patch_size]
        source, target = torch.tensor(img2).float(), torch.tensor(img1).float()
        self.batch_counter += 1
        if self.batch_counter==self.patch_per_image:
            self.read_image()
            self.batch_counter = 0
        if self.noise_type == 'Galsim':
            return util.augment(source, target), math.sqrt(roman.dark_current*140) + roman.read_noise, torch.mean(source).item()
        else:
            return util.augment(source, target), self.gaussian_sample, self.poisson_sample

    
class TestingDataset(Dataset):
    def __init__(self, hf, data_path, image_list, patch_size, scaler, img_channel, noise_type='PG', poisson_settings=20, gaussian_settings=50, exptime_division=False, natural=False, subtract_bkg=False):
        '''
        Dataset class for testing with ground truth frames and added noise.
        '''
        self.rng = np.random.default_rng(1024)
        self.hf = hf
        self.data_path = data_path
        self.image_list = image_list
        self.patch_size = patch_size
        self.scaler = scaler
        self.img_channel = img_channel
        self.natural = natural
        self.exptime_division = exptime_division
        self.subtract_bkg = subtract_bkg
        self.image_size = (4088, 4088) if 'CFHT' not in data_path else (4581, 1024)
        self.num_of_rows, self.num_of_cols = math.ceil(self.image_size[0]/self.patch_size), math.ceil(self.image_size[1]/self.patch_size)
        self.patch_per_image = self.num_of_rows*self.num_of_cols
        self.gaussian_noise_level, self.poisson_noise_level = gaussian_settings, poisson_settings
        self.noise_type = noise_type
        self.batch_counter, self.image_counter = 0, 0
        self.clean_image, self.noisy_image = None, None
        self.read_image()
        
    def __len__(self):
        return self.patch_per_image*len(self.image_list)
         
    def read_image(self):
        '''
        Read and preprocess images from the dataset.
        '''
        self.clean_image = self.hf[self.image_list[self.image_counter]]
        header = json.loads(self.clean_image.attrs['Header'])
        scale_mode = 2 if 'JWST' in self.data_path else 0
        self.clean_image, _, _ = util.read_frame(hf_frame=self.clean_image, scale_mode=scale_mode, noise_type='None', header=header)
        if self.clean_image.shape[0] != 1:
            random_index = random.choice([0, 1])
            self.clean_image = self.clean_image[random_index:random_index+1, :, :]
        if 'JWST' in self.data_path:
            self.current_exptime = header['XPOSURE']
            other_index = 1 - random_index
            self.noisy_image, _, _ = util.read_frame(hf_frame=self.hf[self.image_list[self.image_counter]][other_index:other_index+1, :, :], scale_mode=scale_mode, noise_type='None', header=header)
        elif 'CFHT' in self.data_path:
            self.current_exptime = header['EXPTIME']
            self.noisy_image = util.scale(self.clean_image, self.scaler)[0]
        else:
            self.current_exptime = header['EXPTIME']
            gaussian_sample = self.rng.uniform(0, self.gaussian_noise_level) if self.gaussian_noise_level != None else 0
            poisson_sample = self.rng.uniform(0, self.poisson_noise_level) if self.poisson_noise_level != None else 0
            self.noisy_image = util.add_noise(self.clean_image, gaussian_sample, poisson_sample, self.noise_type, self.rng, header, self.subtract_bkg)

        if self.scaler == 'norm':
            mmscale = MinMaxInterval()
            self.param1, self.param2 = mmscale.get_limits(self.noisy_image)
        elif self.scaler == 'standard':
            sscaler = StandardScaler()
            sscaler.fit(self.noisy_image.reshape(-1, 1))
            self.param1, self.param2 = sscaler.mean_, sscaler.scale_
        else:
            self.param1, self.param2 = 0, 0
        
        self.noisy_image = util.scale(self.noisy_image, self.scaler)[0]

        self.image_counter += 1
        if self.image_counter==len(self.image_list):
            random.shuffle(self.image_list)
            self.image_counter = 0

    def __getitem__(self, idx):
        idx = idx % self.patch_per_image
        row = idx//(self.num_of_cols)
        col = idx%(self.num_of_cols)
        target, source = self.clean_image, self.noisy_image
        if self.exptime_division:
            target /= self.current_exptime
            source /= self.current_exptime
        top, left = row*self.patch_size, col*self.patch_size
        top = (self.image_size[0] - self.patch_size) if (top + self.patch_size) >= self.image_size[0] else top
        left = (self.image_size[1] - self.patch_size) if (left + self.patch_size) >= self.image_size[1] else left
        target = target[:, top:top + self.patch_size, left:left + self.patch_size]
        source = source[:, top:top + self.patch_size, left:left + self.patch_size]
        self.batch_counter += 1
        if self.batch_counter==self.patch_per_image:
            self.read_image()
            self.batch_counter = 0
        source, target = torch.tensor(source).float(), torch.tensor(target).float()
        return source, target, self.param1, self.param2
    





################## Dataloaders for Keck dataset ####################
patch_coordinates = [(2, 2), (128, 2), (2, 258), (128, 258), (2, 514), (128, 514), (384, 388), (510, 388), (384, 518), (510, 518)]

class TrainingDatasetKeck(Dataset):
    def __init__(self, hf, data_path, image_list, patch_size, supervised, scaler, img_channel):
        self.data_path = data_path
        self.hf = hf
        self.image_list = image_list
        self.image_list1, self.image_list2, self.image_list3 = [], [], []
        for file_path in image_list:
            _, header = fitsio.read(file_path, header=True)
            x, y = header['STAR-X'], header['STAR-Y']
            if x < 384 and y < 388:
                self.image_list1.append(file_path)
            elif x < 384 and y >= 388:
                self.image_list2.append(file_path)
            else:
                self.image_list3.append(file_path)
        self.patch_size = patch_size
        self.supervised = supervised
        self.scaler = scaler
        self.img_channel = img_channel
        self.patch_per_image = 10
        self.f1_size, self.f2_size, self.f3_size = len(self.image_list1), len(self.image_list2), len(self.image_list3)
        self.perturbations = self.f1_size*(self.f1_size-1) + self.f2_size*(self.f2_size-1) + self.f3_size*(self.f3_size-1)
        
    def __len__(self):
        return self.patch_per_image*self.perturbations

    def __getitem__(self, idx):
        patch_idx = idx//self.perturbations
        left, top = patch_coordinates[patch_idx]
        idx = idx % self.perturbations
        if idx < self.f1_size*(self.f1_size-1):
            idx1 = idx // (self.f1_size-1)
            idx2 = idx % (self.f1_size-1)
            if idx2 >= idx1:
                idx2 += 1
            image_list = self.image_list1
        elif idx < self.f1_size*(self.f1_size-1) + self.f2_size*(self.f2_size-1):
            idx -= self.f1_size*(self.f1_size-1)
            idx1 = idx // (self.f2_size-1)
            idx2 = idx % (self.f2_size-1)
            if idx2 >= idx1:
                idx2 += 1
            image_list = self.image_list2
        else:
            idx -= self.f1_size*(self.f1_size-1) + self.f2_size*(self.f2_size-1)
            idx1 = idx // (self.f3_size-1)
            idx2 = idx % (self.f3_size-1)
            if idx2 >= idx1:
                idx2 += 1
            image_list = self.image_list3

        img1 = self.hf[image_list[idx1]]
        header1 = json.loads(img1.attrs['Header'])
        img1, _, _ = util.read_frame(hf_frame=img1, scale_mode=2, noise_type='None', header=header1)
        img2 = self.hf[image_list[idx2]]
        header2 = json.loads(img2.attrs['Header'])
        img2, _, _ = util.read_frame(hf_frame=img2, scale_mode=2, noise_type='None', header=header2)

        gaussian_sample = header2['DETRN']/header2['DETGAIN']
        poisson_sample = img2.mean()
        img1 = util.scale(img1, self.scaler)[0]
        img2 = util.scale(img2, self.scaler)[0]
        img1 = img1[:, top:top + self.patch_size, left:left + self.patch_size]
        img2 = img2[:, top:top + self.patch_size, left:left + self.patch_size]
        source, target = torch.tensor(img2).float(), torch.tensor(img1).float()
        return util.augment(source, target), gaussian_sample, poisson_sample

    
class TestingDatasetKeck(Dataset):
    def __init__(self, hf, data_path, image_list, patch_size, scaler, img_channel):
        self.hf = hf
        self.data_path = data_path
        self.image_list = image_list
        self.image_list1, self.image_list2, self.image_list3 = [], [], []
        for file_path in image_list:
            _, header = fitsio.read(file_path, header=True)
            x, y = header['STAR-X'], header['STAR-Y']
            if x < 384 and y < 388:
                self.image_list1.append(file_path)
            elif x < 384 and y >= 388:
                self.image_list2.append(file_path)
            else:
                self.image_list3.append(file_path)
        self.patch_size = patch_size
        self.scaler = scaler
        self.img_channel = img_channel
        self.patch_per_image = 10
        self.f1_size, self.f2_size, self.f3_size = len(self.image_list1), len(self.image_list2), len(self.image_list3)
        self.perturbations = self.f1_size*(self.f1_size-1) + self.f2_size*(self.f2_size-1) + self.f3_size*(self.f3_size-1)
        
    def __len__(self):
        return self.patch_per_image*len(self.image_list)

    def __getitem__(self, idx):
        patch_idx = idx//self.perturbations
        left, top = patch_coordinates[patch_idx]
        idx = idx % self.perturbations
        if idx < self.f1_size*(self.f1_size-1):
            idx1 = idx // (self.f1_size-1)
            idx2 = idx % (self.f1_size-1)
            if idx2 >= idx1:
                idx2 += 1
            image_list = self.image_list1
        elif idx < self.f1_size*(self.f1_size-1) + self.f2_size*(self.f2_size-1):
            idx -= self.f1_size*(self.f1_size-1)
            idx1 = idx // (self.f2_size-1)
            idx2 = idx % (self.f2_size-1)
            if idx2 >= idx1:
                idx2 += 1
            image_list = self.image_list2
        else:
            idx -= self.f1_size*(self.f1_size-1) + self.f2_size*(self.f2_size-1)
            idx1 = idx // (self.f3_size-1)
            idx2 = idx % (self.f3_size-1)
            if idx2 >= idx1:
                idx2 += 1
            image_list = self.image_list3
        img1 = self.hf[image_list[idx1]]
        header1 = json.loads(img1.attrs['Header'])
        img1, _, _ = util.read_frame(hf_frame=img1, scale_mode=2, noise_type='None', header=header1)
        img2 = self.hf[image_list[idx2]]
        header2 = json.loads(img2.attrs['Header'])
        img2, _, _ = util.read_frame(hf_frame=img2, scale_mode=2, noise_type='None', header=header2)


        if self.scaler == 'norm':
            mmscale = MinMaxInterval()
            param1, param2 = mmscale.get_limits(img2)
        elif self.scaler == 'standard':
            sscaler = StandardScaler()
            sscaler.fit(img2.reshape(-1, 1))
            param1, param2 = sscaler.mean_, sscaler.scale_
        else:
            param1, param2 = 0, 0

        img2 = util.scale(img2, self.scaler)[0]
        img1 = img1[:, top:top + self.patch_size, left:left + self.patch_size]
        img2 = img2[:, top:top + self.patch_size, left:left + self.patch_size]
        source, target = torch.tensor(img2).float(), torch.tensor(img1).float()
        return source, target, param1, param2