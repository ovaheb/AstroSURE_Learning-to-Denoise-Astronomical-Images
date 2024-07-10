"""
################## Training by adding noise to GT frame ####################
class TrainingDataset(Dataset):
    def __init__(self, data_path, image_list, patch_size, supervised, scaler, img_channel, natural=False, galsim_noise=False, exptime_division=False):
        '''
        Dataset class for training with ground truth frames and added noise.
        '''
        self.data_path = data_path
        self.image_list = image_list
        self.patch_size = patch_size
        self.supervised = supervised
        self.scaler = scaler
        self.img_channel = img_channel
        self.natural = natural
        self.galsim_noise = galsim_noise
        self.exptime_division = exptime_division
        self.image_size = extract_image_size(os.path.join(self.data_path, self.image_list[0]))
        self.patch_per_image = math.ceil(self.image_size[0]/self.patch_size)*math.ceil(self.image_size[1]/self.patch_size)
        self.gaussian_noise_level, self.poisson_noise_level = 50, 20 #35, 20
        self.batch_counter, self.image_counter = 0, 0
        self.clean_image, self.noisy_image, self.noisy_image2, self.current_exptime = None, None, None, None
        self.read_image()
        
    def __len__(self):
        return self.patch_per_image
            
    def read_image(self):
        '''
        Read and preprocess images from the dataset.
        '''
        #img = fits.open(os.path.join(self.data_path, self.image_list[self.image_counter]), memmap=False)
        #primary_hdu_idx = 0 if 'NOBJS' in img[0].header else 1
        #self.current_exptime = img[primary_hdu_idx].header['exptime']
        #frame = util.read_frame(img, primary_hdu_idx, scale_mode=2)
        img, header = fitsio.read(os.path.join(self.data_path, self.image_list[self.image_counter]), header=True)
        frame = np.expand_dims(np.float32(img), axis=2)
        self.current_exptime = header['exptime']
        self.clean_image = np.transpose(np.clip(frame, 0, MAX_PIXEL_VALUE), (2, 0, 1))
        #img.close()
        gaussian_sample = np.random.uniform(0, self.gaussian_noise_level)
        poisson_sample = np.random.uniform(0, self.poisson_noise_level)
        self.noisy_image = util.scale(util.add_noise(self.clean_image, gaussian_sample, poisson_sample), self.scaler)
        self.noisy_image2 = util.scale(util.add_noise(self.clean_image, gaussian_sample, poisson_sample), self.scaler)
        self.clean_image = util.scale(self.clean_image, self.scaler)
        self.image_counter += 1
        if self.image_counter==len(self.image_list):
            random.shuffle(self.image_list)
            self.image_counter = 0
        
    def __getitem__(self, idx):
        '''
        Get a sample from the dataset.
        '''
        col = idx//(self.image_size[0]//self.patch_size)
        row = idx%(self.image_size[1]//self.patch_size)
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
        return util.augment(source, target)

    
class TestingDataset(Dataset):
    def __init__(self, data_path, image_list, patch_size, scaler, img_channel, natural=False, galsim_noise=False, exptime_division=False):
        '''
        Dataset class for testing with ground truth frames and added noise.
        '''
        self.rng = np.random.default_rng(1024)
        self.data_path = data_path
        self.image_list = image_list
        self.patch_size = patch_size
        self.scaler = scaler
        self.img_channel = img_channel
        self.natural = natural
        self.galsim_noise = galsim_noise
        self.exptime_division = exptime_division
        self.image_size = extract_image_size(os.path.join(self.data_path, self.image_list[0]))
        self.patch_per_image = math.ceil(self.image_size[0]/self.patch_size)*math.ceil(self.image_size[1]/self.patch_size)
        self.gaussian_noise_level, self.poisson_noise_level= 50, 20 #35,20
        self.batch_counter, self.image_counter = 0, 0
        self.clean_image, self.noisy_image, self.current_exptime, self.current_nobjs = None, None, None, None
        self.read_image()
        
    def __len__(self):
        return self.patch_per_image
         
    def read_image(self):
        '''
        Read and preprocess images from the dataset.
        '''
        img = fits.open(os.path.join(self.data_path, self.image_list[self.image_counter]))
        primary_hdu_idx = 0 if 'NOBJS' in img[0].header else 1
        self.clean_image = np.transpose(np.clip(util.read_frame(img, primary_hdu_idx), 0, MAX_PIXEL_VALUE), (2, 0, 1))
        gaussian_sample = np.random.uniform(0, self.gaussian_noise_level)
        poisson_sample = np.random.uniform(0, self.poisson_noise_level)
        self.noisy_image = util.scale(util.add_noise(self.clean_image, gaussian_sample, poisson_sample), self.scaler)
        if self.scaler == 'norm':
            mmscale = MinMaxInterval()
            self.param1, self.param2 = mmscale.get_limits(self.clean_image)
        elif self.scaler == 'standard':
            sscaler = StandardScaler()
            sscaler.fit(img.reshape(-1, 1))
            self.param1, self.param2 = sscaler.mean_, sscaler.scale_
        else:
            self.param1, self.param2 = img[primary_hdu_idx].header['exptime'], img[primary_hdu_idx].header['NOBJS']
        self.clean_image = util.scale(self.clean_image, self.scaler)
        self.image_counter += 1
        if self.image_counter==len(self.image_list):
            random.shuffle(self.image_list)
            self.image_counter = 0
        img.close()
        
    def __getitem__(self, idx):
        col = idx//(self.image_size[0]//self.patch_size)
        row = idx//(self.image_size[1]//self.patch_size)
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
"""




'''
class TrainingDatasetNatural(Dataset):
    def __init__(self, data_path, image_list, patch_size, supervised, img_channel):
        self.data_path = data_path
        self.image_list = image_list
        self.patch_size = patch_size
        self.img_channel = img_channel
        self.supervised = supervised
        self.noise_param = 50
        
    def __len__(self):
        return len(self.image_list)
    
    def augment(self, img1, img2):
        augment_idx = random.randint(0, 5)
        if augment_idx  < 4:
            return tvF.rotate(img1, 90 * augment_idx), tvF.rotate(img2, 90 * augment_idx)
        elif augment_idx == 4:
            return tvF.hflip(img1), tvF.hflip(img2)
        elif augment_idx == 5:
            return tvF.vflip(img1), tvF.vflip(img2)
        return img1, img2
    
    def add_noise(self, img):
        h, w, c = img.shape
        std = np.random.uniform(0,self.noise_param)
        noise = np.random.normal(0, std, (h,w,c))
        noise_img_temp = img + noise
        noise_img = np.clip(noise_img_temp, 0, 255).astype(np.uint8)
        return noise_img
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.data_path, self.image_list[idx])
        img = io.imread(image_name)
        if len(img.shape)!=3:
            image_name = os.path.join(self.data_path, self.image_list[idx+1])
            img = io.imread(image_name)
        h, w, c = img.shape
        if min(h, w) <  self.patch_size:
            img = resize(img, (self.patch_size, self.patch_size), preserve_range=True)
            h, w, c = img.shape
        new_h, new_w = self.patch_size, self.patch_size
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        img1 = img[top:top + new_h, left:left + new_w]
        if not self.supervised:
            img1 = self.add_noise(img1)
        img2 = self.add_noise(img1)
        
        target = tvF.to_tensor(img1.astype(np.float32))
        source = tvF.to_tensor(img2.astype(np.float32))
        return self.augment(source, target)
    
class TestingDatasetNatural(Dataset):
    def __init__(self, test_path, image_list, patch_size, supervised, img_channel):
        self.test_path = test_path
        self.image_list = image_list
        self.patch_size = patch_size
        self.img_channel = img_channel
        self.noise_param = 50
        
    def __len__(self):
        return len(self.image_list)

    def add_noise(self, img):
        h, w, c = img.shape
        std = np.random.uniform(0, self.noise_param)
        noise = np.random.normal(0, std, (h,w,c))
        noise_img_temp = img + noise
        noise_img = np.clip(noise_img_temp, 0, 255).astype(np.uint8)
        return noise_img
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.test_path,self.image_list[idx])
        img = io.imread(image_name)
        if len(img.shape)!=3:
            image_name = os.path.join(self.test_path, self.image_list[idx+1])
            img = io.imread(image_name)
        h, w, c = img.shape
        if min(h, w) <  self.patch_size:
            img = resize(img,(self.patch_size, self.patch_size), preserve_range=True)
            h, w, c = img.shape
        new_h, new_w = self.patch_size, self.patch_size
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        img1 = img[top:top + new_h, left:left + new_w]
        img2 = self.add_noise(img1)
        
        target = tvF.to_tensor(img1.astype(np.float32))
        source = tvF.to_tensor(img2.astype(np.float32))
        return source, target, 0, 0
    
'''