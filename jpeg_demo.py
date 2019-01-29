import torch
import torch.nn as nn
import numpy as np
from model import CAEP
from utils import Kodak,GeneralDS,compute_bpp
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import pytorch_msssim
import time
from torch.utils.data.sampler import SequentialSampler

import numpy as np
from PIL import Image
import subprocess
from tqdm import tqdm
import os
filelist=os.listdir('mixed')
for fichier in filelist[:]: 
    if not(fichier.endswith(".png")):
        filelist.remove(fichier)

import time

times = []
for im in []:#tqdm(filelist):
    start_time = time.time()
    nim = Image.open('mixed/'+im) 
    #print(nim.size)
    nim.save('jpeg/'+im, 'JPEG', quality=100)

    times.append(time.time() - start_time)
    
#print('JPEG: avg time ',np.array(times).mean())

for im in []:#tqdm(filelist):
    start_time = time.time()
    nim = Image.open('mixed/'+im)
    #print(nim.size)
    nim.save('jpeg2000/'+im, 'JPEG2000', quality=100)

    times.append(time.time() - start_time)

#print('JPEG2000: avg time ',np.array(times).mean())
	
d1 = GeneralDS('./mixed/')
dl1 = DataLoader(
    d1,
    batch_size=1,
    sampler=SequentialSampler(d1)
)

d2 = GeneralDS('./jpeg/')
dl2 = DataLoader(
    d2,
    batch_size=1,
    sampler=SequentialSampler(d2)
)

d3 = GeneralDS('./jpeg2000/')
dl3 = DataLoader(
    d3,
    batch_size=1,
    sampler=SequentialSampler(d3)
)

SSIM = pytorch_msssim.SSIM().cuda()

jS = []
j2S = []


for i, atup in enumerate(zip(dl1,dl2)):
   #print(atup[0][0].size())
   im1 = atup[0][0]
   im2 = atup[1][0]
   #print(im1.size())
   jS.append(SSIM(im1,im2).item())
   print(jS[-1])
   #break

print("JPEG: ",np.array(jS).mean())

for i, atup in enumerate(zip(dl1,dl3)):
   im1 = atup[0][0]
   im2 = atup[1][0]
   j2S.append(SSIM(im1,im2).item())
   #break

print("JPEG2000: ",np.array(j2S).mean())

