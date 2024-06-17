import galsim
print(galsim.__version__)
import galsim.roman as roman

print((roman.dark_current*140)**0.5+roman.read_noise)
"""import train as train
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import regex
print(regex.__version__)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             profile_memory=True, record_shapes=True) as prof:
    train.train('')
print(prof.key_averages().table())"""