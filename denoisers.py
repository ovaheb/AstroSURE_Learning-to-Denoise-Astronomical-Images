import numpy as np
import torch
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter
from sklearn.preprocessing import StandardScaler
from utils import utils_image as util
from mask import Masker
from unet_model import UNet, UNet_Upsample
from dncnn_model import DnCNN as DnCNN
import zsn2n as ZSN2N
import bm3d

### Denoiser Classes ###

class UNetDenoiser():
    def __init__(self, model_path, img_channel, device, setting, scaler, dataset_name, train_loss, name, disable_clipping, upsample_mode='bilinear'):
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
            self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
    def to(self, device):
        self.model = self.model.to(device)
        
    def denoise(self, image):
        denoised_image = self.model(image).detach().cpu().numpy()
        denoised_image = np.expand_dims(np.squeeze(denoised_image, axis=1), axis=-1)
        return denoised_image
    
    def summarize(self):
        if self.model_path == 'Random':
            return 'Random UNet Upsample with %d parameters loaded!\n'%sum(p.numel() for p in self.model.parameters())
        else:
            summary = (self.model_path, self.name, sum(p.numel() for p in self.model.parameters()), self.dataset_name, self.setting, self.scaler, self.train_loss)
            return '%s\n%s with %d parameters trained on %s dataset with %s setting, %s scaler, and %s loss loaded!\n'%summary
    
    
class DnCNNDenoiser():
    def __init__(self, model_path, img_channel, device, setting, scaler, dataset_name, train_loss, name, disable_clipping, depth, model_patch_size):
        self.img_channel = img_channel
        self.model_path = model_path
        self.name = 'DnCNN ' + name
        self.device = device
        self.setting = setting
        self.scaler = scaler
        self.dataset_name = dataset_name
        self.train_loss = train_loss
        self.disable_clipping = disable_clipping
        self.depth = depth
        self.model_patch_size = model_patch_size
        self.model = DnCNN(in_nc=img_channel, out_nc=img_channel, nb=17)
        self.load()
    
    def load(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)

    def denoise(self, image):
        denoised_image = self.model(image).detach().cpu().numpy()
        denoised_image = np.expand_dims(np.squeeze(denoised_image), axis=-1)
        return denoised_image
    
    def summarize(self):
        summary = (self.model_path, self.name, sum(p.numel() for p in self.model.parameters()), self.dataset_name, self.setting, self.scaler, self.train_loss)
        return '%s\n%s with %d parameters trained on %s dataset with %s setting, %s scaler, and %s loss loaded!\n'%summary
        
    
class BM3DDenoiser():
    def __init__(self, img_channel):
        self.img_channel = img_channel
        self.VST = False
        self.scaler = 'noscale' if self.VST else 'division'
        self.name = 'BM3D+VST' if self.VST else 'BM3D'
        self.disable_clipping = True
        
    def summarize(self):
        if self.VST:
            return 'BM3D denoiser loaded alongside Variance Stablizing Transformer!\n'
        else:
            return 'BM3D denoiser loaded!\n'
        
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
        
    def summarize(self):
        return 'Zero-shot Noise2Noise model with %d parameters loaded!\n'%sum(map(lambda x: x.numel(), self.model.parameters()))
        
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
    def __init__(self, img_channel, mode, kernel_size=7, gaussian_sigma=2):
        self.name = mode
        self.img_channel = img_channel
        self.scaler = 'noscale'
        self.disable_clipping = True
        self.kernel_size = kernel_size
        self.gaussian_sigma = gaussian_sigma
        
    def denoise(self, image):
        if self.name=='Median Filter':
            return uniform_filter(image.astype(np.float32), size=self.kernel_size)
        elif self.name=='Mean Filter':
            return median_filter(image.astype(np.float32), size=self.kernel_size)
        elif self.name=='Gaussian Filter':
            return gaussian_filter(image.astype(np.float32), sigma=self.gaussian_sigma)

class BaselineDenoiser():
    def __init__(self):
        self.name = 'Baseline'
        self.scaler = 'noscale'
        self.disable_clipping = False

    def summarize(self):
        return self.name
    
    def to(self, device):
        pass

    def denoise(self, image):
        return np.expand_dims(np.squeeze(image.cpu().numpy(), axis=1), axis=-1)
    
if __name__ == '__main__':
    pass