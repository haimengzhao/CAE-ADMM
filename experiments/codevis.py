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

SSIM = pytorch_msssim.SSIM().cuda()

print('Number of GPUs available: ' + str(torch.cuda.device_count()))

model_before = nn.DataParallel(CAEP(15,16).cuda())
pretrained_state_dict = torch.load(f"./chkpt/bpp_10_before_pruning/model.state")
current_state_dict = model_before.state_dict()
current_state_dict.update(pretrained_state_dict)
model_before.load_state_dict(current_state_dict)
model_before.eval()
print('Done Setup Model_before_pruning.')

model_after = nn.DataParallel(CAEP(15,16).cuda())
pretrained_state_dict = torch.load(f"./chkpt/bpp_05/model.state")
#print(pretrained_state_dict)
current_state_dict = model_after.state_dict()
current_state_dict.update(pretrained_state_dict)
model_after.load_state_dict(current_state_dict)
model_after.eval()
print('Done Setup Model_after_pruning.')

dataset = Kodak('/data2/CAE-ADMM/codevis/')

img = dataset[0][0]
patches = dataset[0][1]

stacki = []
for i in range(6):
    stackj = []
    for j in range(4):
        x = torch.Tensor(patches[i, j, :, :, :].unsqueeze(0)).cuda()
        y, c = model_before(x)
        stackj.append(c.squeeze(0).cpu().data)
    stacki.append(torch.cat(stackj, dim=2))
out_before = torch.cat(stacki, dim=1)
out_before = out_before.unsqueeze(1).permute(0, 1, 3, 2)
grid_before = make_grid(out_before, nrow=6, normalize=True)
save_image(grid_before, './codevis/kodim23_code_before.png')

stacki = []
for i in range(6):
    stackj = []
    for j in range(4):
        x = torch.Tensor(patches[i, j, :, :, :].unsqueeze(0)).cuda()
        y, c = model_after(x)
        stackj.append(c.squeeze(0).cpu().data)
    stacki.append(torch.cat(stackj, dim=2))
out_after = torch.cat(stacki, dim=1)
out_after = out_after.unsqueeze(1).permute(0, 1, 3, 2)
grid_after = make_grid(out_after, nrow=6, normalize=True)
save_image(grid_after, './codevis/kodim23_code_after.png')

save_image(torch.Tensor([[0, -1], [1, 0]]), './codevis/test.png')  # confirm that non-positives are black

bsize = 1

# mask those zero elements
def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)


def norm_range(t):
    t = t.clone()
    norm_ip(t, float(t.min()), float(t.max()))
    return t


norm_before = norm_range(out_before)
masked_before = norm_before.masked_fill_(out_before == 0, 0)
save_image(make_grid(masked_before, nrow=6), './codevis/kodim23_masked_code_before.png')

norm_after = norm_range(out_after)
masked_after = norm_before.masked_fill_(out_after == 0, 0)
save_image(make_grid(masked_after, nrow=6), './codevis/kodim23_masked_code_after.png')

whole_dataset = GeneralDS('./Kodak/')
dataloader = DataLoader(
    whole_dataset,
    batch_size=bsize,
    sampler=SequentialSampler(whole_dataset)
)

bpp_before=[]
bpp_after=[]

times=[]
SSIMs=[]
for bi, (img, patches, _) in enumerate(dataloader):
    stacki = []

    #print(img.size())
    for i in range(6):
        stackj = []
        for j in range(4):
            x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
            y, c = model_before(x)
            bpp = compute_bpp(c, x.shape[0], 'crop', save=False)
            #SSIMs.append(SSIM(x, y))
            #print(bpp)
            bpp_before.append(bpp)

            stackj.append(c.cpu().data)
        stacki.append(torch.cat(stackj, dim=3))
    avg_out_before = torch.cat(stacki, dim=2)

    #stacki = []

    #start_time = time.time()
    time_im = 0
 
    imi = []
    for i in range(6):
        #stackj = []
        imj = []
        for j in range(4):

            start_time = time.time()
            x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
            y, c = model_after(x)
 
            
            time_im += time.time() - start_time
            bpp = compute_bpp(c, x.shape[0], 'crop', save=False)
            #SSIMs.append(SSIM(x, y).item())
            #print(bpp)
            bpp_after.append(bpp)
          
            imj.append(y.squeeze(0).cpu().data)
        imi.append(torch.cat(imj, dim=2))
    im = torch.cat(imi, dim=1).unsqueeze(0)
    
    #print(im.size())
    #img = img.unsqueeze(1).permute(0, 1, 3, 2)
    #print(img.size())
    times.append(time_im/img.size()[0])
    #print(time_im)
    SSIMs.append(SSIM(im, img).item())
    print(SSIMs[-1])
            #stackj.append(c.cpu().data)
        #stacki.append(torch.cat(stackj, dim=3))
    #avg_out_after = torch.cat(stacki, dim=2)

# ratio of zero elements before pruning
#r_before_p = torch.sum(avg_out_before == 0).numpy() / torch.numel(avg_out_before)
#print('Ratio of zero elements before pruning : %.2f%%' % (r_before_p * 100))

# ratio of zero elements after pruning
#r_after_p = torch.sum(avg_out_after == 0).numpy() / torch.numel(avg_out_after)
#print('Ratio of zero elements after pruning : %.2f%%' % (r_after_p * 100))

# the increased ratio of zeros
#relative_change = (r_after_p - r_before_p) / r_before_p
#print('The increased ratio of zeros : %.2f %%' % (relative_change * 100))

# the decreased ratio of non-zeros
#relative_change_non_zero = ((1 - r_after_p) - (1 - r_before_p)) / (1 - r_before_p)
#print('The decreased ratio of non-zeros : %.2f %%' % (relative_change_non_zero * 100))

print('bpp after ',np.array(bpp_after).mean())

print('bpp before ',np.array(bpp_before).mean())

print('avg SSIM ',np.array(SSIMs).mean())

print('avg time ',np.array(times).mean())
