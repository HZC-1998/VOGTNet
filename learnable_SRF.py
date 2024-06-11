import numpy as np
import os
import xlrd
import torch
import torch.nn as nn
class SpeNet(nn.Module):
    def __init__(self, msi_channels, PAN_channel):
        super(SpeNet, self).__init__()
        self.PAN_channel = PAN_channel
        self.msi_channels = msi_channels
        self.conv2d_list = nn.Conv2d(self.msi_channels,self.PAN_channel,1,1,0,bias=False)
    def forward(self, input):
        layer=self.conv2d_list
        out = layer(input).div_(layer.weight.data.sum(dim=1).view(1))
        return out.clamp_(0,1)
