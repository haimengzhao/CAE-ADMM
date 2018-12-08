from torchvision.utils import save_image
from torch import nn
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from huffmancoding import huffman_encode, huffman_decode


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


class Quantizer(torch.autograd.Function):
    def forward(ctx, x):
        r = torch.rand(x.shape)
        p = x
        eps = torch.zeros(x.shape)
        eps[r <= p] = (1 - x)[r <= p]
        eps[r > p] = (-x)[r > p]
        y = x + eps
        return y

    def backward(ctx, grad_outputs):
        return grad_outputs


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
