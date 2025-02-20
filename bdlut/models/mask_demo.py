from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *
from utils import bit_plane_slicing, floor_func


class Mask_demo(nn.Module):
    def __init__(self, nf=64, upscale=4, act=nn.ReLU):
        super(Mask_demo, self).__init__()
        self.act = act()
        self.upscale = upscale
        # mask
        self.MSB_mask = torch.nn.Parameter(torch.ones(1, 1, 5, 5))
        self.LSB_mask = torch.nn.Parameter(torch.ones(1, 1, 5, 5))
        self.MSB_mask.requires_grad = False
        self.LSB_mask.requires_grad = False
        
        # MSB
        self.msb_conv1 = Conv(1, nf, 5, stride=1, padding=2, dilation=1)
        self.msb_conv2 = ActConv(nf, nf, 1, act=act)
        self.msb_conv3 = ActConv(nf, nf, 1, act=act)
        self.msb_conv4 = ActConv(nf, nf, 1, act=act)
        self.msb_conv5 = ActConv(nf, nf, 1, act=act)
        self.msb_conv6 = Conv(nf, upscale * upscale, 1)
        self.msb_module = nn.Sequential(self.msb_conv1, self.msb_conv2, self.msb_conv3, self.msb_conv4, self.msb_conv5, self.msb_conv6)
        
        # LSB
        self.lsb_conv1 = Conv(1, nf, 5, stride=1, padding=2, dilation=1)
        self.lsb_conv2 = ActConv(nf, nf, 1, act=act)
        self.lsb_conv3 = ActConv(nf, nf, 1, act=act)
        self.lsb_conv4 = ActConv(nf, nf, 1, act=act)
        self.lsb_conv5 = ActConv(nf, nf, 1, act=act)
        self.lsb_conv6 = Conv(nf, upscale * upscale, 1)
        self.lsb_module = nn.Sequential(self.lsb_conv1, self.lsb_conv2, self.lsb_conv3, self.lsb_conv4, self.lsb_conv5, self.lsb_conv6)

        # pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        
    
    def forward(self, img_lr):
        B, C, H, W = img_lr.size()
        # Prepare inputs for two branches
        img_lr = img_lr.reshape(B*C, 1, H, W)
        batch_L255 = torch.floor(img_lr * 255)
        MSB, LSB = bit_plane_slicing(batch_L255, bit_mask='11110000')
        MSB = MSB / 255.0
        LSB = LSB / 255.0
        
        self.msb_conv1.conv.weight.data = self.msb_conv1.conv.weight.data * self.MSB_mask
        self.lsb_conv1.conv.weight.data = self.lsb_conv1.conv.weight.data * self.LSB_mask
        
        bias = 127.0
        # MSB
        MSB_out = 0.0
        for r in [0,1,2,3]:
            batch = self.msb_module(torch.rot90(MSB, r, [2, 3]))
            batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
            MSB_out += floor_func(batch)

        MSB_out = MSB_out / 3 / 255.
        MSB_out = torch.clamp(MSB_out, -1, 1)

        # LSB
        LSB_out = 0.0
        for r in [0,1,2,3]:
            batch = self.lsb_module(torch.rot90(LSB, r, [2, 3]))
            batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
            LSB_out += floor_func(batch)
        LSB_out = LSB_out / 2 / 255.
        LSB_out = torch.clamp(LSB_out, -1, 1)

        output = MSB_out + LSB_out
        output = self.pixel_shuffle(output)

        output = output.reshape(B*C, 1, self.upscale*(H), self.upscale*(W))
        
        output += nn.Upsample(scale_factor=self.upscale, mode='nearest')(img_lr)
        
        output = output.reshape(B, C, self.upscale*(H), self.upscale*(W))

        return torch.clamp(output, 0, 1)
        
        