import torch
import torch.nn as nn
import numpy as np
from model import CAEP
from utils import Kodak
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

print('Number of GPUs available: ' + str(torch.cuda.device_count()))

model_before = nn.DataParallel(CAEP(15).cuda())
pretrained_state_dict = torch.load(f"./chkpt/bpp_10_before_pruning/model.state")
current_state_dict = model_before.state_dict()
current_state_dict.update(pretrained_state_dict)
model_before.load_state_dict(current_state_dict)
model_before.eval()
print('Done Setup Model_before_pruning.')

model_after = nn.DataParallel(CAEP(15).cuda())
pretrained_state_dict = torch.load(f"./chkpt/bpp_10/model.state")
current_state_dict = model_after.state_dict()
current_state_dict.update(pretrained_state_dict)
model_after.load_state_dict(current_state_dict)
model_after.eval()
print('Done Setup Model_after_pruning.')

dataset = Kodak('./codevis/')

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

whole_dataset = Kodak('./Kodak/')
dataloader = DataLoader(
    whole_dataset,
    batch_size=whole_dataset.__len__()
)

for bi, (img, patches, _) in enumerate(dataloader):
    stacki = []
    for i in range(6):
        stackj = []
        for j in range(4):
            x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
            y, c = model_before(x)
            stackj.append(c.cpu().data)
        stacki.append(torch.cat(stackj, dim=3))
    avg_out_before = torch.cat(stacki, dim=2)

    stacki = []
    for i in range(6):
        stackj = []
        for j in range(4):
            x = torch.Tensor(patches[:, i, j, :, :, :]).cuda()
            y, c = model_after(x)
            stackj.append(c.cpu().data)
        stacki.append(torch.cat(stackj, dim=3))
    avg_out_after = torch.cat(stacki, dim=2)

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
