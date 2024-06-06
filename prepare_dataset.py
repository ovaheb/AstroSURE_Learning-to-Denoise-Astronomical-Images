import h5py
import numpy as np
import os
from utils import utils_image as util
from astropy.io import fits
import fitsio
import io
from PIL import Image
import json
from pathlib import Path
import math

def prepare_dataset(dataset_path, hf_root=os.environ.get('SLURM_TMPDIR'), force=False):
    hf_path = os.path.join(hf_root, dataset_path.split('/')[-1])
    if os.path.exists(hf_path):
        if force:
            os.remove(hf_path)
        else:
            return hf_path
    with h5py.File(hf_path, 'w', libver='latest') as hf:
        for file_path in Path(dataset_path).rglob('*'):
            file_path = str(file_path)
            if util.is_fits_file(file_path):
                file_name = file_path.split('/')[-1]
                with fitsio.FITS(file_path) as fits_file:
                    try:
                        img = np.float32(fits_file['SCI'].read())
                        header = dict(fits_file['SCI'].read_header())
                        header['VAR_RNOISE'] = math.sqrt(np.nanmean(fits_file['VAR_RNOISE'].read()[0, :, :]))
                        header['VAR_POISSON'] = math.sqrt(np.nanmean(fits_file['VAR_POISSON'].read()[0, :, :]))
                    except:
                        img = np.float32(fits_file[1].read())
                        header = dict(fits_file[1].read_header())
                    try:
                        DQ_frame = fits_file['DQ'].read()
                        DQ_frame, _, _ = util.read_frame(hf_frame=DQ_frame, scale_mode=2)
                    except:
                        DQ_frame = None

                frame, _, _ = util.read_frame(hf_frame=img, scale_mode=2)
                if DQ_frame is not None:
                    data = np.concatenate((frame, DQ_frame), axis=0)
                else:
                    data = frame
                dataset = hf.create_dataset(file_path, data=data)
                dataset.attrs['Header'] = json.dumps(header)
            if util.is_image_file(file_path):
                with open(file_path, 'rb') as img:
                     data = img.read()
                data = np.asarray(data)
                _ = hf.create_dataset(file_name, data=data)
        hf.swmr_mode = True
    print('HDF5 file size: %d MB'%int(os.path.getsize(hf_path)//1024**2))
    hf.close()
    return hf_path