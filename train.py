import warnings
import sys
import wandb
import os
import h5py
from utils import utils_image as util
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import numpy as np
import torchvision.transforms.functional as tvF
from unet_model import UNet, UNet_Upsample
from dncnn_model import DnCNN as net
import time
from dataset import TrainingDataset, TestingDataset
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from astropy.visualization import PercentileInterval, ZScaleInterval, MinMaxInterval, BaseInterval
import random
import math
import torch.distributed as dist
from mask import Masker
from transforms import GroupTransform
from prepare_dataset import prepare_dataset
import datetime
from pathlib import Path
import json

#@profile
### Main training script ###
def train(argv):
    torch.backends.cudnn.benchmark = True
    
    args = parse_args(argv)
    ### Specific training Poilicies for different SSL schemes
    if args.data_path.split("/")[-1]=='JWST':
        unsupervised = True 
        if args.supervised=='N2C':
            args.supervised = 'N2N'
            print("JWST data is unsupervised, switching to N2N training!")
        args.noise_type = None
        args.disable_early_stopping = True
        args.disable_clipping = True
    else:
        unsupervised = False
    if args.supervised=='N2Sa':
        args.loss = 'L2'
        args.scale = 'standard'
        args.J_invariant_reconstruction = None
        print("N2Sa training is only possible with L2 loss and standard scaling!")
    elif args.supervised=='N2Se':
        args.invariance_strength = None
    else:
        args.mask_width = None
        args.masking_scheme = None
        args.J_invariant_reconstruction = None
        args.invariance_strength = None
    if args.supervised!='EI':
        args.EI_transforms = None
        args.EI_strength = None
    if args.supervised=='SURE':
        args.scale = 'noscale'
    else:
        args.SURE_tau = None
        args.SURE_tau2 = None
    if args.architecture!='UNet-Upsample': 
        args.upsample_mode = None
    if args.noise_type=='Galsim':
        args.poisson_settings = None
        args.gaussian_settings = None
        if args.subtract_bkg == False:
            args.subtract_bkg = True
            print("Training onGalsim noise is only possible with background subtraction!")
    else:
        args.subtract_bkg = None
        if args.noise_type=='P':
            args.gaussian_settings = 0
        elif args.noise_type=='G':
            args.poisson_settings = 0
        elif args.noise_type=='None':
            args.poisson_settings = None
            args.gaussian_settings = None

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_name = args.supervised + '_' + args.data_path.split("/")[-1] + '_' + args.scale 
    run_name += '_' + args.loss + '_'
    if args.architecture == 'UNet-Upsample':
        run_name += args.architecture + '-' + args.upsample_mode + '_'
    else:
        run_name += args.architecture + '_'
    if args.noise_type in ['None', 'Galsim']:
        run_name += args.noise_type
    else:
        if 'P' in args.noise_type:
            run_name += 'P' + str(args.poisson_settings) 
        if 'G' in args.noise_type:
            run_name += 'G' + str(args.gaussian_settings)
    if args.noise_type == 'Galsim':
        if args.subtract_bkg:
            run_name += '-nobkg'
        else:
            run_name += '-bkg'
    if args.disable_clipping:
        run_name += '_noclip'
    else:
        run_name += '_clip'

    run_name += '_' + str(date)
    
    if args.enable_logging:
        run = initialize_wandb(args, run_name, date)
        print_model_parameters()
        print(run.dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = args.checkpoint_path + '/' + run_name
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    
    hf_path = prepare_dataset(args.data_path)
    hf = h5py.File(hf_path, 'r', swmr=True)
    ## Define the data path and train-test split
    file_list = [str(file) for file in Path(args.data_path).rglob('*') if (util.is_image_file(str(file)) or util.is_fits_file(str(file)))]
    dataset_file_length = len(file_list)
    if unsupervised:
        train_image_list = [file for file in file_list if json.loads(hf[file].attrs['Header'])['RA_V1']<=52.9642]
        val_image_list = [file for file in file_list if file not in train_image_list]
        train_file_length = len(train_image_list)
        val_file_length = dataset_file_length - train_file_length
    else:
        train_file_length = int(dataset_file_length * 0.8)
        val_file_length = dataset_file_length - train_file_length
        random.seed(7)
        train_image_list = random.sample(file_list, train_file_length)
        random.seed(None)
        val_image_list = [file for file in file_list if file not in train_image_list]
    

    
    train_dataset = TrainingDataset(hf, args.data_path, train_image_list, args.patch_size, args.supervised, args.scale, args.img_channel, args.noise_type, args.poisson_settings, args.gaussian_settings, args.exptime_division, args.natural, args.subtract_bkg)
    val_dataset = TestingDataset(hf, args.data_path, val_image_list, args.patch_size, args.scale, args.img_channel, args.noise_type, args.poisson_settings, args.gaussian_settings, args.exptime_division, args.natural, args.subtract_bkg)
    
    validation_batch_size = 1
    
    train_length, val_length = len(train_dataset), len(val_dataset)
    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, sampler=sampler, generator=torch.Generator().manual_seed(1024))
    val_loader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False, pin_memory=True, num_workers=4)
    print("Data is loaded! Training: %d patches, Validation: %d patches"%(train_length, val_length))
    
    # Define the model
    if args.architecture == 'UNet':
        model = UNet(in_channels=args.img_channel, out_channels=args.img_channel, load_from=args.load_from + '/best_model.pth')
    elif args.architecture == 'UNet-Upsample':
        model = UNet_Upsample(in_channels=args.img_channel, out_channels=args.img_channel, mode=args.upsample_mode, load_from=args.load_from + '/best_model.pth')
    elif args.architecture == 'DnCNN':
        model = net.DnCNN()
    
    # Define the functions specific to each SSL training scheme
    if args.supervised=='N2V':
        masker = None # TO DO!
    elif args.supervised in ['N2Se', 'N2Sa']:
        masker = Masker(width = args.mask_width, mode=args.masking_scheme)
    elif args.supervised=='EI':
        Tg = GroupTransform(args.EI_transforms)
    
    # Define the loss function
    if args.loss == 'L2':
        criterion = nn.MSELoss().to(device)
    elif args.loss == 'L1':
        criterion = nn.L1Loss().to(device)
    else:
        loss_weight, prior_weight = 1.0, 0.5
        if args.enable_logging:
            wandb.config["Loss, Prior weights"] = (loss_weight, prior_weight)
        if args.loss == 'L1-L1Prior':
            criterion = util.CustomLoss(loss_weight=loss_weight, prior_weight=prior_weight, loss=nn.L1Loss())
        elif args.loss == 'L2-L1Prior':
            criterion = util.CustomLoss(loss_weight=loss_weight, prior_weight=prior_weight, loss=nn.MSELoss())

    if args.fix_learning_rate:
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    elif args.supervised=='EI':
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8, amsgrad=False)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold_mode='rel', threshold=0.001, factor=0.5, patience=100)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, amsgrad=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold_mode='rel', threshold=0.001, factor=0.5, patience=20, min_lr=0.0001)
    
    epochs = args.epochs

    ## Move model to the GPU
    model = model.to(device, memory_format=torch.channels_last)
    print('Total number of model parameters: %d'%sum(p.numel() for p in model.parameters()))
    if next(model.parameters()).device.type == 'cuda':
        print("Model is on GPU!") # Check if we are on GPU
    
    ## Defining training helper variables
    save_per_epoch, patience = 20, 50
    patience_idx = patience
    best_val_loss = float('inf')
    #training_loss, val_loss = [], []
    counter_train, counter_validation = 0, 0
    #grad_scaler = torch.cuda.amp.GradScaler(growth_interval=200)
    
    if args.natural:
        min_pixel, max_pixel = 0.0, 256.0
    elif unsupervised:
        min_pixel, max_pixel = -20.0, 238.0
    else:

            min_pixel, max_pixel = 0.0, 65536.0
    ## Trainng loop
    print("Starting Training ...")
    epoch_best_model = -1
    since = time.time()
    for epoch in tqdm(range(epochs), leave=False, colour="green"):
        epoch_train_loss = 0.0
        epoch_train_loss_term1, epoch_train_loss_term2, epoch_train_loss_term3 = 0.0, 0.0, 0.0
        model.train()
        for batch_idx, ((source, target), param1, param2) in enumerate(train_loader):
            counter_train += args.batch_size
            optimizer.zero_grad()
            if args.supervised=='N2Sa':
                source = source.to(device)
                masked_source, mask = masker.mask(source, batch_idx)
                #with torch.cuda.amp.autocast():
                denoised_source_raw = model(source)
                denoised_source_masked = model(masked_source)
                loss_mean = nn.MSELoss()
                loss_sum = nn.MSELoss(reduction = 'sum')
                l_rec = loss_mean(denoised_source_raw, source)
                l_inv = loss_sum(denoised_source_raw*mask, denoised_source_masked*mask) / torch.sum(mask)
                loss = l_rec + args.invariance_strength * math.sqrt(l_inv)
            elif args.supervised in ['N2Se', 'N2V']:
                source = source.to(device)
                masked_source, mask = masker.mask(source, batch_idx)
                #with torch.cuda.amp.autocast():
                denoised_source = model(masked_source)
                loss = criterion(denoised_source*mask, source*mask)
            elif args.supervised=='EI':
                source = source.to(device)
                x1 = model(source) # reconstruction of the groundtruth x
                x2 = Tg.apply(x1) # transform
                #with torch.cuda.amp.autocast():
                x3 = model(x2) # reconstruction of the transformed data
                loss = criterion(x1, source) + args.EI_strength * criterion(x3, x2)
                
            elif args.supervised=='SURE':
                source = source.to(device)
                #with torch.cuda.amp.autocast():
                denoised_source = model(source)
                if args.noise_type=='G':
                    if args.scale=='norm':
                        param1 = param1 / max_pixel
                        tau = args.SURE_tau
                    else:
                        tau = args.SURE_tau * max_pixel
                    param1 = param1.to(device)
                    b = torch.randn_like(source).to(device)

                    mse = (denoised_source - source).pow(2).reshape(source.size(0), -1).mean(1)
                    offset = param1**2
                    div = 2 * param1**2 * (b * (model(source + b * tau) - denoised_source) / tau).reshape(source.size(0), -1).mean(1)
                    loss = (mse - offset + div).mean()
                
                elif args.noise_type=='P':
                    tau = args.SURE_tau * max_pixel
                    param2 = param2.view(args.batch_size, 1, 1, 1).to(device)
                    b = torch.rand_like(source) > 0.5
                    b = (2*b - 1) * 1.0
                    b = b.to(device)

                    mse = (denoised_source - source).pow(2).reshape(source.size(0), -1).mean(1)
                    offset = (param2 * source).reshape(source.size(0), -1).mean(1)
                    div = (2.0 / tau * (b * source * param2 * (model(source + tau * b) - denoised_source))).reshape(source.size(0), -1).mean(1)
                    loss = (mse - offset + div).mean()

                elif args.noise_type in ['PG', 'Galsim', 'None']:
                    tau1, tau2 = args.SURE_tau * max_pixel, args.SURE_tau2 * max_pixel
                    param1 = param1.view(args.batch_size, 1, 1, 1).to(device)
                    param2 = param2.view(args.batch_size, 1, 1, 1).to(device)
                    b = torch.rand_like(source) > 0.5
                    b = (2 * b - 1) * 1.0 
                    p = 0.7236 #0.5 + 0.5*np.sqrt(1/5)
                    c = torch.ones_like(b) * np.sqrt(p / (1 - p))
                    c[torch.rand_like(c) < p] = -np.sqrt((1 - p) / p)
                    b = b.to(device)
                    c = c.to(device)

                    meas1 = denoised_source
                    meas2 = model(source + tau1 * b)
                    meas2p = model(source + tau2 * c)
                    meas2n = model(source - tau2 * c)

                    mse = (meas1 - source).pow(2).reshape(source.size(0), -1).mean(1)
                    loss_div1 = (2 / tau1 * ((b * (param2 * source + param1**2)) * (meas2 - meas1))).reshape(source.size(0), -1).mean(1)
                    offset = param2 * source.reshape(source.size(0), -1).mean(1) + param1**2
                    loss_div2 = (-2 * param1**2 * param2 / (tau2**2) * (c * (meas2p + meas2n - 2 * meas1))).reshape(source.size(0), -1).mean(1)
                    div = loss_div1 + loss_div2
                    loss = (mse - offset + div).mean()
            else:
                target = target.to(device)
                source = source.to(device, memory_format=torch.channels_last)
                #with torch.cuda.amp.autocast():
                denoised_source = model(source)
                loss = criterion(denoised_source, target)

            

            loss.backward()
            optimizer.step()
            
            #grad_scaler.scale(loss).backward()
            #grad_scaler.unscale_(optimizer)  # unscale to clip gradient
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #grad_scaler.step(optimizer)
            #grad_scaler.update()
            #if grad_scaler._scale < 128:
            #    grad_scaler._scale = torch.tensor(128).to(grad_scaler._scale)
            epoch_train_loss += loss.item()*args.batch_size
            if args.supervised == 'N2Sa':
                epoch_train_loss_term1 += l_rec.item()*args.batch_size
                epoch_train_loss_term2 += math.sqrt(l_inv)*args.batch_size
            elif args.supervised == 'EI':
                epoch_train_loss_term1 += criterion(x1, source).item()*args.batch_size
                epoch_train_loss_term2 += criterion(x3, x2).item()*args.batch_size
            elif args.supervised == 'SURE':
                epoch_train_loss_term1 += mse.mean().item()*args.batch_size
                epoch_train_loss_term2 += offset.mean().item()*args.batch_size
                epoch_train_loss_term3 += div.mean().item()*args.batch_size
        # print('Target ', [torch.min(target), torch.max(target)])
        # print('Source ', [torch.min(source), torch.max(source)])
        # print('Denoised ', [torch.min(denoised_source), torch.max(denoised_source)])
        # Calculate validation metrics for early stopping
        model.eval()
        epoch_val_loss, epoch_val_l1, epoch_val_l2, epoch_val_psnr, epoch_val_kl = 0.0, 0.0, 0.0, 0.0, 0.0
        if epoch==0:
            epoch_val_loss_baseline, epoch_val_l1_baseline, epoch_val_l2_baseline, epoch_val_psnr_baseline, epoch_val_kl_baseline = 0.0, 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (source, target, param1, param2) in enumerate(val_loader):
                counter_validation += validation_batch_size
                if args.supervised=='N2Se' and args.J_invariant_reconstruction:
                    denoised_source = masker.infer_full_image(source.to(device, non_blocking=True), model)
                else:
                    denoised_source = model(source.to(device, non_blocking=True))
                denoised_source = denoised_source.detach().cpu()

                if args.natural:
                    denoised_source *= 256
                else:
                    denoised_source = util.descale(denoised_source, args.scale, param1, param2)
                
                if not args.disable_clipping:
                    denoised_source = torch.clamp(denoised_source, min_pixel, max_pixel)
                epoch_val_loss += criterion(target, denoised_source).item()
                epoch_val_l1 += torch.mean(torch.abs(target - denoised_source)).item()
                epoch_mse = torch.mean(torch.pow(target - denoised_source, 2))
                epoch_val_l2 += epoch_mse.item()
                epoch_val_psnr += 20 * torch.log10(max_pixel / torch.sqrt(epoch_mse)).item()
                epoch_val_kl += util.kl_divergence(torch.ravel(target), torch.ravel(denoised_source), is_torch=True).item()
                if epoch==0:
                    source = util.descale(source, args.scale, param1, param2)
                    epoch_val_loss_baseline += criterion(target, source).item()
                    epoch_val_l1_baseline += torch.mean(torch.abs(target - source)).item()
                    epoch_mse = torch.mean(torch.pow(target - source, 2))
                    epoch_val_l2_baseline += epoch_mse.item()
                    epoch_val_psnr_baseline += 20 * torch.log10(max_pixel / torch.sqrt(epoch_mse)).item()
                    epoch_val_kl_baseline += util.kl_divergence(torch.ravel(target), torch.ravel(source), is_torch=True).item()
        # print('Target ', [torch.min(target), torch.max(target)])
        # print('Source ', [torch.min(source), torch.max(source)])
        # print('Denoised ', [torch.min(denoised_source), torch.max(denoised_source)])
        if args.fix_learning_rate:
            scheduler.step()
        else:
            scheduler.step(epoch_val_loss / val_length)
        torch.cuda.empty_cache()
        # Save the first instance of the model metrics
        if epoch==0:       
            best_val_loss, best_val_l1, best_val_l2, best_val_psnr, best_val_kl = epoch_val_loss, epoch_val_l1, epoch_val_l2, epoch_val_psnr, epoch_val_kl
        
        # Early stop if val loss does not improve 1% after 20 epochs
        if epoch_val_loss < 0.999*best_val_loss:
            best_val_loss, best_val_l1, best_val_l2, best_val_psnr, best_val_kl = epoch_val_loss, epoch_val_l1, epoch_val_l2, epoch_val_psnr, epoch_val_kl
            patience_idx = patience
            epoch_best_model = epoch + 1
            torch.save(model.state_dict(), checkpoint_path + '/best_model.pth')
            
        elif not args.disable_early_stopping:
            patience_idx -= 1
            if patience_idx == 0:
                print("Early stopping: No improvement in validation loss for %d epochs!"%patience)
                break
        ## Log the metrics
        #training_loss.append(epoch_train_loss / train_length)
        #val_loss.append(epoch_val_loss / val_length)   
        if args.enable_logging: 
            wandb.log({"Training Loss": epoch_train_loss / train_length,
                        "Validation Loss": epoch_val_loss / val_length,
                        "Validation L1": epoch_val_l1 / val_length,
                        "Validation L2": epoch_val_l2 / val_length,
                        "Validation PSNR": epoch_val_psnr / val_length,
                        "Validation KL Divergence": epoch_val_kl / val_length,
                        "Validation Loss Baseline": epoch_val_loss_baseline / val_length,
                        "Validation L1 Loss Baseline": epoch_val_l1_baseline / val_length,
                        "Validation L2 Loss Baseline": epoch_val_l2_baseline / val_length,
                        "Validation PSNR Baseline": epoch_val_psnr_baseline / val_length,
                        "Validation KL Divergence Baseline": epoch_val_kl_baseline / val_length,
                        "Learning Rate": optimizer.param_groups[0]['lr']})
            if args.supervised in ['N2Sa', 'EI', 'SURE']:
                wandb.log({ "Training Loss Term 1": epoch_train_loss_term1 / train_length,
                            "Training Loss Term 2": epoch_train_loss_term2 / train_length,
                            "Training Loss Term 3": epoch_train_loss_term3 / train_length})
        # Save checkpoints from training in case sth goes wrong
        if epoch%save_per_epoch==0:
            save_model(model, epoch + 1, checkpoint_path)
            print('Checkpoint at epoch {}; Train loss: {}; Validation loss: {}'.format(epoch + 1, epoch_train_loss / train_length, epoch_val_loss / val_length)) 
    

    
    time_elapsed = time.time() - since      
    print('Training completed in {} epochs; {} train iterations(batch={}) and {} validation iteration; {:.0f}m {:.0f}s'.format(epoch + 1, counter_train, args.batch_size, counter_validation, time_elapsed // 60, time_elapsed % 60)) 
    print('Lowest validation loss achieved at epoch %d'%epoch_best_model)
    if args.enable_logging:
        wandb.config["Number of Epochs"] = epoch + 1
        wandb.log({"Validation Loss Final": best_val_loss / val_length,
        "Validation L1 Final": best_val_l1 / val_length,
        "Validation L2 Final": best_val_l2 / val_length,
        "Validation PSNR Final": best_val_psnr / val_length,
        "Validation KL Divergence Final": best_val_kl / val_length,
        "Runtime": int(time_elapsed // 60),})

    hf.close()
    del train_dataset, val_dataset, model
    if args.enable_logging:
        wandb.finish()
        if args.wandb_mode=='offline':
            with open('/home/ovaheb/scratch/jobs_to_sync.txt', 'a') as file:
                file.write(run.dir + "\n")
        
### Helper Functions ###
def save_model(model, epoch, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(model.state_dict(), checkpoint_path + '/denoise_epoch_{}.pth'.format(epoch))
    
def print_model_parameters():
    print("Experiment Parameters:")
    for key, value in wandb.config.items():
        print(f"{key}: {value}")

def initialize_wandb(args, run_name, date):
    return wandb.init(project="n2n", name=run_name, config={
                    "GPU": torch.cuda.get_device_name(0),
                    "Architecture": args.architecture,
                    "Dataset": args.data_path.split("/")[-1],
                    "Patch Size": args.patch_size,
                    "Supervised": args.supervised,
                    "Noise Type": args.noise_type,
                    "Poisson Settings": args.poisson_settings,
                    "Gaussian Settings": args.gaussian_settings,
                    "Loss": args.loss,
                    "Scale": args.scale if not args.natural else None,
                    "Divide by Exposure Time": args.exptime_division if not args.natural else None,
                    "Datetime": date,
                    "Learning Rate": args.lr,
                    "Number of Channels": args.img_channel,
                    "Batch Size": args.batch_size,
                    "Upsample Mode": args.upsample_mode,
                    "J-Invariant Reconstruction": args.J_invariant_reconstruction,
                    "Mask Width": args.mask_width,
                    "Masking Scheme": args.masking_scheme,
                    "Subtract Background": args.subtract_bkg,
                    "EI strength": args.EI_strength,
                    "Invariance Strength": args.invariance_strength,
                    "EI transforms": args.EI_transforms,
                    "SURE Tau": args.SURE_tau,
                    "SURE Tau2": args.SURE_tau2,
                    "Disable clipping": args.disable_clipping,
                    "Fix Learning Rate": args.fix_learning_rate,
                    "Pretrained weights loaded from": args.load_from}, mode=args.wandb_mode,)
    
## Parse arguments to train images with different settings and parameters
def parse_args(argv):
    parser = argparse.ArgumentParser(prog='training', add_help=True)
    parser.add_argument('--data_path', type=str, default='/home/ovaheb/projects/def-sdraper/ovaheb/simulated_data/trainsets/medium_radius')
    parser.add_argument('--checkpoint_path', type=str, default='/home/ovaheb/scratch/temp/checkpoints')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--architecture', type=str, default='UNet-Upsample', help='UNet/UNet-Upsample/DnCNN')
    parser.add_argument('--upsample_mode', type=str, default='bilinear', help='nearest/bilinear/bicubic')
    parser.add_argument('--loss', type=str, default='L2', help='L2/L1/L2_L1Prior/L1_L1Prior')
    parser.add_argument('--scale', type=str, default='noscale', help='noscale/norm/standard/division/arcsinh/anscombe')
    parser.add_argument('--supervised', type=str, default='N2C', help='N2C/N2N/N2Se/N2Sa/S2S/EI/SURE/REI')
    parser.add_argument('--noise_type', type=str, default='PG', help='P/G/PG/Galsim/None')
    parser.add_argument('--poisson_settings', type=int, default=20)
    parser.add_argument('--gaussian_settings', type=int, default=50)
    parser.add_argument('--subtract_bkg', type=bool, default=False)
    parser.add_argument('--exptime_division', type=bool, default=False)
    parser.add_argument('--J_invariant_reconstruction', type=bool, default=False)
    parser.add_argument('--mask_width', type=int, default=4)
    parser.add_argument('--masking_scheme', type=str, default='interpolate', help='interpolate/zero')
    parser.add_argument('--invariance_strength', type=float, default=2)
    parser.add_argument('--EI_transforms', type=str, default='S', help='(S)hift/(R)otate/(F)ourier')
    parser.add_argument('--EI_strength', type=int, default=1)
    parser.add_argument('--SURE_tau', type=float, default=0.0001)
    parser.add_argument('--SURE_tau2', type=float, default=0.001)
    parser.add_argument('--enable_logging', type=bool, default=False)
    parser.add_argument('--disable_early_stopping', type=bool, default=False)
    parser.add_argument('--wandb_mode', type=str, default='offline', help='online/offline')
    parser.add_argument('--disable_clipping', type=bool, default=False)
    parser.add_argument('--fix_learning_rate', type=bool, default=False)
    parser.add_argument('--load_from', type=str, default=None, help='Path to the .pth file containig weights of the previously trained model')
    args = parser.parse_args(argv)
    return args

if __name__=="__main__":
    train(sys.argv[1:])
    
    
    
'''
#For visualization of the inputs going inside the network
mmscale = MinMaxInterval()
pscale = PercentileInterval(99.0)
mm, maskm = masker.mask(source, batch_idx)
print(mmscale.get_limits(source[0,:,:]), mmscale.get_limits(mm[0,:,:]))
fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey='all')
image = axs[0].imshow(pscale(np.squeeze(source[0,:,:].numpy())), interpolation='nearest', cmap='gray')
cmap = image.get_cmap()
axs[0].set_title('Image')
axs[0].set_yticks([])
axs[1].imshow(pscale(np.squeeze(mm[0,:,:].numpy())), interpolation='nearest', cmap=cmap)
axs[1].set_title('Input')
axs[1].set_yticks([])
axs[2].imshow(pscale(np.squeeze((source[0,:,:]*maskm).numpy())), interpolation='nearest', cmap=cmap)
axs[2].set_title('Target')
axs[2].set_yticks([]) 
plt.tight_layout()
plt.show()
if counter_train>=5:
    break''' 