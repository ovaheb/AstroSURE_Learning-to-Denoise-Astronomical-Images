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
import random
import sep
from tqdm import tqdm

def prepare_dataset(dataset_path, hf_root=os.environ.get('SLURM_TMPDIR'), force=False, method='nearest', bias=0.0):
    if 'CFHT' in dataset_path:
        hf_root = '/home/ovaheb/scratch'
    hf_path = os.path.join(hf_root, dataset_path.split('/')[-1])
    if os.path.exists(hf_path):
        if force:
            os.remove(hf_path)
        else:
            return hf_path
    with h5py.File(hf_path, 'w', libver='latest') as hf:
        file_list = [str(file) for file in Path(dataset_path).rglob('*') if (util.is_image_file(str(file)) or util.is_fits_file(str(file)))]
        if 'CFHT' in dataset_path:
            random.seed(7)
            file_list = random.sample(file_list, 20)
            random.seed(None)
        for file_path in tqdm(file_list):
            if util.is_fits_file(file_path):
                file_name = file_path.split('/')[-1]
                with fitsio.FITS(file_path) as fits_file:
                    if 'JWST' in dataset_path:
                        img = np.float32(fits_file['SCI'].read())
                        header = dict(fits_file['SCI'].read_header())
                        header['VAR_RNOISE'] = math.sqrt(np.nanmean(fits_file['VAR_RNOISE'].read()[0, :, :]))
                        header['VAR_POISSON'] = math.sqrt(np.nanmean(fits_file['VAR_POISSON'].read()[0, :, :]))
                        DQ_frame = fits_file['DQ'].read()
                        DQ_frame, _, _ = util.read_frame(hf_frame=DQ_frame, scale_mode=2)
                        frame, _, _ = util.read_frame(hf_frame=img, scale_mode=2)
                        data = np.concatenate((frame, DQ_frame), axis=0)
                        
                    elif 'keck' in dataset_path:
                        img = np.float32(fits_file[0].read())
                        data, _, _ = util.read_frame(hf_frame=img, scale_mode=2)
                        header = dict(fits_file[0].read_header())

                    elif 'CFHT' in dataset_path:
                        for hdu_index, hdu in enumerate(fits_file[1:]):
                            frame = np.float32(hdu.read())
                            headerA, headerB = dict(hdu.read_header()), dict(hdu.read_header())
                            imgA, imgB = frame[3:-33, 32:1056], frame[3:-33, 1056:-32]
                            imgA, imgB = util.remove_nan_CCD(imgA, method=method), util.remove_nan_CCD(imgB, method=method)
                            imgA, imgB = util.normalize_CCD_range(imgA, bias), util.normalize_CCD_range(imgB, bias)
                            readoutA, readoutB = headerA['RDNOISEA']/headerA['GAINA'], headerB['RDNOISEB']/headerB['GAINB']
                            headerA['gaussian'], headerB['gaussian'] = readoutA, readoutB
                            headerA['poisson'], headerB['poisson'] = float(sep.Background(imgA.astype(np.float64)).globalrms), float(sep.Background(imgB.astype(np.float64)).globalrms)

                            datasetA = hf.create_dataset(f"{file_path}{hdu_index+1:02d}A", data=imgA)
                            datasetA.attrs['Header'] = json.dumps(headerA)
                            datasetB = hf.create_dataset(f"{file_path}{hdu_index+1:02d}B", data=imgB)
                            datasetB.attrs['Header'] = json.dumps(headerB)

                    else:
                        img = np.float32(fits_file[1].read())
                        header = dict(fits_file[1].read_header())
                        data, _, _ = util.read_frame(hf_frame=img, scale_mode=2)

                if 'CFHT' not in dataset_path:
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