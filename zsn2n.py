import numpy as np 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## Defining the network architecture and forward function
class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x

# Definition of the downsampler which creates two downsample images from 8*8 windows
def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2

# Mean squared loss error used during training
def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

# Loss function including consisteny loss and residual loss
def loss_func(noisy_img, model):
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)
    loss_res = 1/2*(mse(noisy1, pred2)+mse(noisy2, pred1))
    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons=1/2*(mse(pred1, denoised1) + mse(pred2, denoised2))
    loss = loss_res + loss_cons
    return loss

# The function that trains a network on two downsampled images from one noisy image using MSE
def train(model, optimizer, noisy_img):
    loss = loss_func(noisy_img, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to denoise an input image
def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
    return pred 