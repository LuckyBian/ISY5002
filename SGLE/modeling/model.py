import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Traditional Convolution
class TC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TC, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, input):
        out = self.conv(input)
        return out



# Depthwise Separable Convolution
class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DSC, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class enhance_net_nopool(nn.Module):
    def __init__(self, scale_factor, conv_type='dsc'):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

        # Define Conv type
        if conv_type == 'dsc':
            self.conv = DSC
        elif conv_type == 'dc':
            self.conv = DC
        elif conv_type == 'tc':
            self.conv = TC
        else:
            print("conv type is not available")

        # Encoder
        self.e_conv1 = self.conv(3, number_f)
        self.e_conv2 = self.conv(number_f, number_f)
        self.e_conv3 = self.conv(number_f, number_f)
        self.e_conv4 = self.conv(number_f, number_f)

        # Middle
        self.middle_conv = self.conv(number_f, number_f)

        # Decoder
        self.d_conv1 = self.conv(number_f * 2, number_f)
        self.d_conv2 = self.conv(number_f * 2, number_f)
        self.d_conv3 = self.conv(number_f * 2, 3)

    def enhance(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x):
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode='bilinear')

        # Encoder
        e1 = self.relu(self.e_conv1(x_down))
        e2 = self.relu(self.e_conv2(e1))
        e3 = self.relu(self.e_conv3(e2))
        e4 = self.relu(self.e_conv4(e3))

        # Middle
        middle = self.relu(self.middle_conv(e4))

        # Decoder
        d1 = self.relu(self.d_conv1(torch.cat([e3, middle], 1)))
        d2 = self.relu(self.d_conv2(torch.cat([e2, d1], 1)))
        x_r = F.tanh(self.d_conv3(torch.cat([e1, d2], 1)))

        if self.scale_factor != 1:
            x_r = self.upsample(x_r)

        # Enhancement
        enhance_image = self.enhance(x, x_r)

        return enhance_image, x_r



