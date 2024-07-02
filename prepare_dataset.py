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

def prepare_dataset(dataset_path, hf_root=os.environ.get('SLURM_TMPDIR'), force=False, method='nearest', bias=0.0):
    hf_path = os.path.join(hf_root, dataset_path.split('/')[-1])
    if os.path.exists(hf_path):
        if force:
            os.remove(hf_path)
        else:
            return hf_path
    with h5py.File(hf_path, 'w', libver='latest') as hf:
        file_list = [str(file) for file in Path(dataset_path).rglob('*') if (util.is_image_file(str(file)) or util.is_fits_file(str(file)))]
        for file_path in file_list:
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
                        for hdu in fits_file[1:]:
                            frame = np.float32(hdu.read())
                            headerA, headerB = dict(hdu.read_header()), dict(hdu.read_header())
                            header = headerA
                            imgA, imgB = frame[30:-33, 32:1056], frame[30:-33, 1056:-32]
                            imgA, imgB = util.remove_nan(imgA, method=method), util.remove_nan(imgB, method=method)
                            imgA, imgB = util.normalize_CCD_range(imgA, bias), util.normalize_CCD_range(imgB, bias)
                            readoutA, readoutB = header['RDNOISEA']/header['GAINA'], header['RDNOISEB']/header['GAINB']
                            darkA = math.sqrt(header['DARKCUR']*(header['EXPTIME']/3600.0)/header['GAINA'])
                            darkB = math.sqrt(header['DARKCUR']*(header['EXPTIME']/3600.0)/header['GAINB'])
                            headerA['gaussian'], headerB['gaussian'] = readoutA + darkA, readoutB + darkB
                            headerA['poisson'], headerB['poisson'] = float(np.mean(imgA)), float(np.mean(imgB))

                    else:
                        img = np.float32(fits_file[1].read())
                        header = dict(fits_file[1].read_header())
                        data, _, _ = util.read_frame(hf_frame=img, scale_mode=2)

                if 'CFHT' not in dataset_path:
                    dataset = hf.create_dataset(file_path, data=data)
                    dataset.attrs['Header'] = json.dumps(header)
                else:
                    datasetA = hf.create_dataset(file_path + 'A', data=imgA)
                    datasetA.attrs['Header'] = json.dumps(headerA)
                    datasetB = hf.create_dataset(file_path + 'B', data=imgB)
                    datasetB.attrs['Header'] = json.dumps(headerB)

            if util.is_image_file(file_path):
                with open(file_path, 'rb') as img:
                     data = img.read()
                data = np.asarray(data)
                _ = hf.create_dataset(file_name, data=data)
        hf.swmr_mode = True
    print('HDF5 file size: %d MB'%int(os.path.getsize(hf_path)//1024**2))
    hf.close()
    return hf_path