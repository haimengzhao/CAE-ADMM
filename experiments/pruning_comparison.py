import torch
import torch.nn as nn
import numpy as np
from model import CAEP
from utils import Kodak,GeneralDS,compute_bpp,save_kodak_img
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import pytorch_msssim
import time
from torch.utils.data.sampler import SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

SSIM = pytorch_msssim.SSIM().cuda()

print('Number of GPUs available: ' + str(torch.cuda.device_count()))

model_before = nn.DataParallel(CAEP(15,32).cuda())
pretrained_state_dict = torch.load(f"./chkpt/bpp_10_before_pruning/model.state")
current_state_dict = model_before.state_dict()
current_state_dict.update(pretrained_state_dict)
model_before.load_state_dict(current_state_dict)
model_before.eval()
print('Done Setup Model_before_pruning.')

model_after = nn.DataParallel(CAEP(15,32).cuda())
pretrained_state_dict = torch.load(f"./chkpt/bpp_10/model.state")
#print(pretrained_state_dict)
current_state_dict = model_after.state_dict()
current_state_dict.update(pretrained_state_dict)
model_after.load_state_dict(current_state_dict)
model_after.eval()
print('Done Setup Model_after_pruning.')

bsize = 1

# mask those zero elements
def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t):
    t = t.clone()
    norm_ip(t, float(t.min()), float(t.max()))
    return t

whole_dataset = GeneralDS('./mixed/')
dataloader = DataLoader(
    whole_dataset,
    batch_size=bsize,
    sampler=SequentialSampler(whole_dataset)
)

writer = SummaryWriter(log_dir=f'./')

bpp_before=[]
bpp_after=[]

for bi, (img, patches, _) in tqdm(enumerate(dataloader)):
    bpp_im = 0
    stacki = []
    for i in range(6):
        stackj = []
        for j in range(4):
            x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
            y, c = model_before(x)
            bpp = compute_bpp(c, x.shape[0], 'crop', save=False)
            bpp_im += bpp

            stackj.append(c.cpu().data)
        stacki.append(torch.cat(stackj, dim=3))
    avg_out_before = torch.cat(stacki, dim=2)
    bpp_before.append(bpp_im/24)

    bpp_im = 0
    stacki = []
    for i in range(6):
        stackj = []
        for j in range(4):
            x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
            y, c = model_after(x)
            
            bpp = compute_bpp(c, x.shape[0], 'crop', save=False)
            bpp_im += bpp
            stackj.append(c.cpu().data)
        stacki.append(torch.cat(stackj, dim=3))
    avg_out_after = torch.cat(stacki, dim=2)
    bpp_after.append(bpp_im/24)
# ratio of zero elements before pruning
r_before_p = torch.sum(avg_out_before == 0).numpy() / torch.numel(avg_out_before)
print('Ratio of zero elements before pruning : %.2f%%' % (r_before_p * 100))

# ratio of zero elements after pruning
r_after_p = torch.sum(avg_out_after == 0).numpy() / torch.numel(avg_out_after)
print('Ratio of zero elements after pruning : %.2f%%' % (r_after_p * 100))

# the increased ratio of zeros
relative_change = (r_after_p - r_before_p) / r_before_p
print('The increased ratio of zeros : %.2f %%' % (relative_change * 100))

# the decreased ratio of non-zeros
relative_change_non_zero = ((1 - r_after_p) - (1 - r_before_p)) / (1 - r_before_p)
print('The decreased ratio of non-zeros : %.2f %%' % (relative_change_non_zero * 100))

bpp_b = np.array(bpp_before)
bpp_a = np.array(bpp_after)

print('bpp before: mu {} std {}'.format(bpp_b.mean(), bpp_b.std()))

print('bpp after: mu {} std {}'.format(bpp_a.mean(), bpp_a.std()))
