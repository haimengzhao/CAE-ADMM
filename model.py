import torch
from torch import nn
from utils import conv_downsample, conv_same, res_layers, Bottleneck, sub_pix, quantize, MaskedPruner


class CAEP(nn.Module):
    def __init__(self, num_resblocks):
        super(CAEP, self).__init__()
        self.num_resblocks = num_resblocks
        self.threshold = torch.Tensor([1e-5])
        self.prune = False

        # Encoder
        self.E_Conv_1 = conv_downsample(3, 64)  # 3,128,123 => 64,64,64
        self.E_PReLU_1 = nn.PReLU()
        self.E_Conv_2 = conv_downsample(64, 128)  # 64,64,64 => 128,32,32
        self.E_PReLU_2 = nn.PReLU()
        self.E_Res = res_layers(128, num_blocks=self.num_resblocks)
        self.E_Conv_3 = conv_downsample(128, 64)  # 128,32,32 => 64,16,16

        # max_bpp = 64*16*16/128/128 = 1

        # Decoder
        self.D_SubPix_1 = sub_pix(64, 128, 2)  # 64,16,16 => 128,32,32
        self.D_PReLU_1 = nn.PReLU()
        self.D_Res = res_layers(128, num_blocks=self.num_resblocks)
        self.D_SubPix_2 = sub_pix(128, 64, 2)  # 128,32,32 => 64,64,64
        self.D_PReLU_2 = nn.PReLU()
        self.D_SubPix_3 = sub_pix(64, 3, 2)  # 64,64,64 => 3,128,128
        self.tanh = nn.Tanh()

        self.__init_parameters__()

    def __init_parameters__(self):
        # Initialize Parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, Bottleneck):
                nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.conv2.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.conv3.weight, mode='fan_out')
                nn.init.constant_(m.bn3.weight, 0)

    def prune_mode(self, prune=True, threshold=1e-5):
        self.prune = prune
        self.threshold = torch.Tensor([threshold])
        self.Pruner = MaskedPruner()

    def forward(self, x):
        x = self.E_Conv_1(x)
        x = self.E_PReLU_1(x)
        x = self.E_Conv_2(x)
        x = self.E_PReLU_2(x)
        x = self.E_Res(x)
        x = self.E_Conv_3(x)

        if self.prune:
            x = self.Pruner(x, self.threshold)
        x = quantize(x)

        y = self.D_SubPix_1(x)
        y = self.D_PReLU_1(y)
        y = self.D_Res(y)
        y = self.D_SubPix_2(y)
        y = self.D_PReLU_2(y)
        y = self.D_SubPix_3(y)
        y = (self.tanh(y) + 1) / 2

        return y, x
