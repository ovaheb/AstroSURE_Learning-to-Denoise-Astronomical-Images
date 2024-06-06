import torch
import random
import numpy as np
import torchvision.transforms.functional as tvF

class Shift():
    def __init__(self):
        self.n_trans = 3
        self.max_offset = 0
        
    def apply(self, x):
        H, W = x.shape[-2], x.shape[-1]
        shifts_row = random.sample(list(np.concatenate([-1*np.arange(1, H), np.arange(1, H)])), self.n_trans)
        shifts_col = random.sample(list(np.concatenate([-1*np.arange(1, W), np.arange(1, W)])), self.n_trans)
        x = torch.cat([torch.roll(x, shifts=[sx, sy], dims=[-2,-1]).type_as(x) for sx, sy in zip(shifts_row, shifts_col)], dim=0)
        return x


class Rotate():
    def __init__(self):
        self.n_trans = 4
        
    def apply(self, x):
        idx_list = np.arange(1, 6)
        return torch.cat([(tvF.hflip(x) if idx==4 else tvF.vflip(x)) if idx>=4 else tvF.rotate(x, int(idx*90)) for idx in idx_list], dim=0)
    
    
class GroupTransform():
    def __init__(self, EI_transforms):
        self.EI_transforms = EI_transforms
        self.Tg = []
        self.n_trans = 0
        if 'R' in EI_transforms:
            self.Tg.append(Rotate())
            self.n_trans += 4
        if 'S' in EI_transforms:
            self.Tg.append(Shift())
            self.n_trans += 3
        if 'F' in EI_transforms:
            pass
        
    def apply(self, x):
        data = []
        for T in self.Tg:
            data.append(T.apply(x))
        data = torch.cat((data), dim=0)
        return data