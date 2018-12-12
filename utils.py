from torchvision.utils import save_image
from torch import nn
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np
from huffmancoding import huffman_encode, huffman_decode
from torchvision import transforms
import sys
from collections import Counter
<<<<<<< HEAD
<<<<<<< HEAD
from range_coder import RangeEncoder, prob_to_cum_freq
import os
=======

>>>>>>> parent of 56c932c... Replace huffman with range encoder.
=======

>>>>>>> parent of 56c932c... Replace huffman with range encoder.

def save_imgs(imgs, to_size, name):
    imgs = imgs.view(imgs.size(0), *to_size)
    save_image(imgs, name)


def conv_downsample(in_planes, out_planes):
    return nn.Sequential(
        nn.ReflectionPad2d(2),
        nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=2, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def conv_same(in_planes, out_planes):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def sub_pix(in_planes, out_planes, upscale_factor):
    mid_planes = upscale_factor**2 * out_planes
    return nn.Sequential(
        nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(mid_planes),
        nn.PixelShuffle(upscale_factor)
    )


def quantize(x):
    with torch.no_grad():
        floor = x.floor()
        r = torch.rand(x.shape).cuda()
        p = x - floor
        eps = torch.zeros(x.shape).cuda()
        eps[r <= p] = (floor + 1 - x)[r <= p]
        eps[r > p] = (floor - x)[r > p]
    y = x + eps
    return y


class MaskedPruner(torch.autograd.Function):
    def forward(ctx, input, threshold):
        threshold.requires_grad = False
        x = input.masked_fill_(torch.abs(input) <= threshold, 0)
        return x

    def backward(ctx, grad_output):
        return grad_output


class Bottleneck(nn.Module):
    def __init__(self, planes):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu1 = nn.RReLU(inplace=True)
        self.relu2 = nn.RReLU(inplace=True)
        self.relu3 = nn.RReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)

        return out


def res_layers(planes, num_blocks, ):
    return nn.Sequential(*[Bottleneck(planes) for i in range(num_blocks)])


class BSDS500Crop128(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = Image.open(path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)


class Kodak(Dataset):
    def __init__(self, folder_path):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]
        img = np.array(Image.open(path))
        if img.shape[0] != 768:
            img = img.transpose((1, 0, 2))
        h, w, c = img.shape

        img = img / 255.0

        # (768,512)--> 6*4 128*128

        patches = np.array(np.split(np.array(np.split(img, 6, axis=0)), 4, axis=2))
        patches = np.transpose(patches, (1, 0, 4, 2, 3))

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        patches = torch.from_numpy(patches).float()

        return img, patches, path

    def get_random(self):
        i = np.random.randint(0, len(self.files))
        return self[i]

    def __len__(self):
        return len(self.files)


# def compute_bpp(code, batch_size, prefix, dir='./code/', save=True):
def compute_bpp(code, batch_size):
    # Huffman coding
    # Not Effective
    c = list(code.data.cpu().numpy().astype(np.int32).flatten())
    # tree_size, data_size = huffman_encode(c, prefix, save_dir=dir, save=save)
    # bpp = (tree_size + data_size) / batch_size / 128 / 128 * 8
<<<<<<< HEAD
<<<<<<< HEAD
    # print(rle(c))7
    prob = np.array(list(Counter(c).values())) / len(c)
    cumFreq = prob_to_cum_freq(prob)
    encoder = RangeEncoder(f'./code/{name}.txt')
    print(cumFreq)
    # print(c)
    encoder.encode(c, cumFreq)
    encoder.close()
    bpp = os.stat(f'./code/{name}.txt').st_size * 8 / 128 / 128 / batch_size
=======
=======
>>>>>>> parent of 56c932c... Replace huffman with range encoder.
    # print(rle(c))
    print(len(c))
    print(Counter(c))
    sys.setrecursionlimit(10000000)
    bpp = rle(c) / batch_size / 128 / 128
<<<<<<< HEAD
>>>>>>> parent of 56c932c... Replace huffman with range encoder.
=======
>>>>>>> parent of 56c932c... Replace huffman with range encoder.
    return bpp


def save_kodak_img(model, img, index, patches, writer, ei):
    # save a Kodak img
    stacki = []
    for i in range(6):
        stackj = []
        for j in range(4):
            x = torch.Tensor(patches[index, i, j, :, :, :].unsqueeze(0)).cuda()
            y, c = model(x)
            stackj.append(y.squeeze(0).cpu().data)
        stacki.append(torch.cat(stackj, dim=2))
    out = torch.cat(stacki, dim=1)
    y = torch.cat((img[index], out), dim=2).unsqueeze(0)
    save_imgs(imgs=y, to_size=(3, 768, 2 * 512),
              name=f"./output/test_{index}_{ei}.png")
    writer.add_image(f'img/test_{index}', y, ei)
