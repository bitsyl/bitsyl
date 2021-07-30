import numpy as np
import torch
from layers import ConvOffset2D
from torch.nn import functional as F
import torch.nn as nn


# CGCconv
class CConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if kernel_size == 1 :
            self.ind = True
        else:
            self.ind = False
            self.oc = out_channels
            self.ks = kernel_size

            # the target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))

            # the dimension of the latent repsentation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)

            # the context encoding module
            self.ce = nn.Linear(ws * ws, self.num_lat, False)
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            # activation function is relu
            self.act = nn.ReLU(inplace=True)

            # the number of groups in the channel interacting module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            # the channel interacting module
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            # the gate decoding module
            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)

            # used to prrepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            # sigmoid function
            self.sig = nn.Sigmoid()

    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()
            weight = self.weight
            # allocate glbal information
            gl = self.avg_pool(x).view(b, c, -1)
            # context-encoding module
            out = self.ce(gl)
            # use different bn for the following two branches
            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)
            # gate decoding branch 1
            out = self.gd(out)
            # channel interacting module
            if self.g > 3:
                # grouped linear
                oc = self.ci(self.act(self.ci_bn2(ce2). \
                                      view(b, c // self.g, self.g, -1).transpose(2, 3))).transpose(2, 3).contiguous()
            else:
                # linear layer for resnet.conv1
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2, 1))).transpose(2, 1).contiguous()
            oc = oc.view(b, self.oc, -1)
            oc = self.ci_bn(oc)
            oc = self.act(oc)
            # gate decoding branch 2
            oc = self.gd2(oc)
            # produce gate
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
            # currently only handle square input and output
            return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))


def cconv3x3(in_planes, out_planes, stride=1):
    """3x3 context gated convolution with padding"""
    return CConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def cconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return CConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SplitCGCconv(nn.Module):

    def __init__(self,inplanes,planes,kernel_size,radix=4,stride=1):
        super(SplitCGCconv, self).__init__()
        self.kernel_size = kernel_size
        self.radix = radix
        self.inplanes = inplanes
        self.stride = stride
        self.offset = ConvOffset2D(inplanes)
        self.conv3x3 = cconv3x3(inplanes//radix,planes//radix,stride)
        self.conv1x1 = cconv1x1(inplanes//radix,planes//radix)

    def forward(self,x):
        x = self.offset(x)
        splited = torch.split(x, self.inplanes//self.radix, dim=1)
        if self.kernel_size == 3:
            out = [self.conv3x3(i) for i in splited]
            return torch.cat(out,1)
        elif self.kernel_size == 1:
            out = [self.conv1x1(i) for i in splited]
            return torch.cat(out,1)

