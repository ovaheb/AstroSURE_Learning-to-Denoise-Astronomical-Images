import os.path
import numpy as np
import logging
import argparse
import importlib
import utils.utils_image as util
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
from utils import utils_logger
from astropy.visualization import PercentileInterval, MinMaxInterval, ZScaleInterval
import plotly.express as px
import seaborn as sns
from astropy.visualization import hist
from matplotlib.patches import Ellipse

def show_fits(img_path, compact = True):
    plot_size = 12
    
    MMinterval = MinMaxInterval()
    ZSinterval = ZScaleInterval()
    percentile = 99.
    Pinterval = PercentileInterval(percentile)
    cmap = False
    
    if not util.is_fits_file(img_path.split('/')[-1]):
        raise Exception('Wrong format!')
        
    plt.rcParams.update({'font.size': 10})
    fits_img = fits.open(img_path, uint=True)
    primary_hdu_idx = 0 if 'NOBJS' in fits_img[0].header else 1
    hdr = fits_img[primary_hdu_idx].header
    print("Filters: %s\nGround truth image has %d objects with the exposure time of %d." %(hdr['filters'], hdr['nobjs'], hdr['exptime']))
    fits_img.info()
    n_hdu = 1 if compact else len(fits_img)
    table = fits_img[-1].data
    for index in range(n_hdu):
        frame = util.read_frame(fits_img, primary_hdu_idx + index)
        if frame is None:
            continue
        z_frame = ZSinterval(frame)
        mm_frame = MMinterval(frame)
        p_frame = Pinterval(frame)

        ### Images ###
        fig, axs = plt.subplots(1, 2, figsize=(2*plot_size, plot_size), sharey='all')
        if not cmap:
            image = axs[0].imshow(np.squeeze(mm_frame), interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
            cmap = image.get_cmap()
            print('Colormap is set!')
        else:
            image = axs[0].imshow(np.squeeze(mm_frame), interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(image, fraction=0.046, pad=0.04)
        axs[0].set_title('Raw Image (MinMax Scale)')
        image = axs[1].imshow(np.squeeze(z_frame), interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(image, fraction=0.046, pad=0.04)
        axs[1].set_title('Scaled Image (%.1f%% Percentile)'%percentile)
        
        if 'Object Dimension' in table.dtype.names:
            for i in range(len(table)):
                e = Ellipse(xy=(table['Object X'][i], table['Object Y'][i]), width=table['Object Dimension'][i],
                                height=table['Object Dimension'][i], angle=0)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                axs[1].add_artist(e)
        '''# ZScale #
        image = axs[2].imshow(np.squeeze(z_frame), interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
        #fig.colorbar(image, fraction=0.046, pad=0.04)
        axs[2].set_title('Image after Zscale')'''
        plt.subplots_adjust(wspace=0.1)
        plt.show()
        
        
        ### Histograms ###
        bins = 30
        fig, axs = plt.subplots(1, 2, figsize=(2*plot_size, plot_size), sharey='all')
        axs[0].hist(np.ravel(frame), bins=bins, histtype='stepfilled', alpha=0.75)
        axs[0].set_ylim(1e0,2e7)
        axs[0].set_yscale('log')
        axs[0].set_title('Histogram of raw pixel values')
        axs[1].hist(np.ravel(p_frame), bins=bins, histtype='stepfilled', alpha=0.75)
        axs[1].set_yscale('log')
        axs[1].set_title('Histogram of pixel values after %.1f%% Percentile'% percentile)
        '''axs[2].hist(np.ravel(z_frame), bins=bins, histtype='stepfilled', alpha=0.75)
        axs[2].set_yscale('log')
        axs[2].set_title('Histogram of pixel values after Zscale')'''
        plt.subplots_adjust(wspace=0.1)
        plt.show()

    fits_img.close()

def summarize_fits_dataset(data_path, structured_noise=False):
    paths = util.get_image_paths(data_path)
    img_list = []
    for idx, img in enumerate(paths):
        if not util.is_fits_file(img):
            continue
        fits_img = fits.open(img)
        
        if 'NOBJS' in fits_img[0].header:
            primary_hdu_idx = 0
        else:
            primary_hdu_idx = 1
        hdr = fits_img[0 + primary_hdu_idx].header
        img_list.append((hdr['exptime'], hdr['NOBJS'], idx))
    
    percentile = 99.
    Pinterval = PercentileInterval(percentile)
    
    obj_sorted = sorted(img_list, key=lambda x:x[1])
    mid_idx = len(obj_sorted)//2
    low_nobj = sorted(obj_sorted[:11], key=lambda x:x[0])
    medium_nobj = sorted(obj_sorted[mid_idx - 5:mid_idx + 6], key=lambda x:x[0])
    high_nobj = sorted(obj_sorted[-11:], key=lambda x:x[0])
    nobj_list = [low_nobj, medium_nobj, high_nobj]
    plt.rcParams.update({'font.size': 10})
    for hdu_idx in range(2):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                idx = nobj_list[i][j*3][2]
                
                fits_img = fits.open(paths[idx])
                frame = util.read_frame(fits_img, primary_hdu_idx + hdu_idx, structured_noise=False if hdu_idx==0 else structured_noise)
                if frame is None:
                    continue
                image = ax.imshow(np.squeeze(Pinterval(frame)), interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('NObjs=%d, Exptime=%d'%(nobj_list[i][j*5][1], nobj_list[i][j*5][0]))
                if i==0:
                    if j==0: ax.set_title('Low Exposure Time')
                    if j==1: ax.set_title('Medium Exposure Time')
                    if j==2: ax.set_title('High Exposure Time')
                    
                if j==0:
                    if i==0: ax.set_ylabel('Low Number of Objs')
                    if i==1: ax.set_ylabel('Medium Number of Objs')
                    if i==2: ax.set_ylabel('High Number of Objs')
        plt.tight_layout()
        plt.savefig('/arc/home/ovaheb/results/expVSobj_' + data_path.split("/")[-1] + str(hdu_idx) + '.png', dpi=600)
        plt.show()
        
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            idx = nobj_list[i][j*3][2]

            fits_img = fits.open(paths[idx])
            frame = util.read_frame(fits_img, primary_hdu_idx, structured_noise=False if hdu_idx==0 else structured_noise)
            if frame is None:
                continue
            ax.hist(np.ravel(frame), bins=30, histtype='step', color='red', log=True)
            frame = util.read_frame(fits_img, primary_hdu_idx + 1, structured_noise=False if hdu_idx==0 else structured_noise)
            if frame is None:
                continue
            ax.hist(np.ravel(frame), bins=30, histtype='step', color='blue', log=True)
            
            
            ax.set_xlabel('NObjs=%d, Exptime=%d'%(nobj_list[i][j*5][1], nobj_list[i][j*5][0]))
            if i==0:
                if j==0: ax.set_title('Low Exposure Time')
                if j==1: ax.set_title('Medium Exposure Time')
                if j==2: ax.set_title('High Exposure Time')
            else:
                ax.set_xticks([])

            if j==0:
                if i==0: ax.set_ylabel('Low Number of Objs')
                if i==1: ax.set_ylabel('Medium Number of Objs')
                if i==2: ax.set_ylabel('High Number of Objs')
            else:
                ax.set_yticks([])
                
    plt.tight_layout()
    plt.savefig('/arc/home/ovaheb/results/hists_' + data_path.split("/")[-1] + '.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    show_fits(sys.argv[1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''### ECDFs ###
fig, axs = plt.subplots(1, 3, figsize=(3*plot_size, plot_size), sharey='all')
sorted_data = np.sort(np.ravel(frame))
ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axs[0].plot(sorted_data, ecdf, marker='.', linestyle='none')
axs[0].set_xlabel('Pixel Value')
axs[0].set_title('Empirical CDF of raw pixel values')
axs[0].margins(0.02)
sorted_data = np.sort(np.ravel(z_frame))
ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axs[1].plot(sorted_data, ecdf, marker='.', linestyle='none')
axs[1].set_xlabel('Pixel Value')
axs[1].set_title('Empirical CDF of pixel values after Zscale')
axs[1].margins(0.02)
sorted_data = np.sort(np.ravel(p_frame))
ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
axs[2].plot(sorted_data, ecdf, marker='.', linestyle='none')
axs[2].set_xlabel('Pixel Value')
axs[2].set_title('Empirical CDF of pixel values after %.1f%% Percentile'% percentile)
axs[2].margins(0.02)
plt.subplots_adjust(wspace=0.1)
plt.show()'''