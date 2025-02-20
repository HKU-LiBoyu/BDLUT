import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bit_plane_slicing, decode_bit_mask
from .luts import HDLUT, HDVLUT, HDBLUT, HDBVLUT, HLLUT, HSLUT, HDBLLUT, HDBTLUT
from torchvision.utils import save_image

class HKLUT(nn.Module): 
    def __init__(self, msb_weights, lsb_weights, msb='hdb', lsb='hdv', upscale=2):
        super(HKLUT, self).__init__()
        self.upscale = upscale
        self.bit_mask = '11110000'
        self.msb_bits, self.lsb_bits, self.msb_step, self.lsb_step = decode_bit_mask(self.bit_mask)
        unit_dict = {'hd': HDLUT, 'hl': HLLUT, 'hs': HSLUT, 'hdv': HDVLUT, 'hdb': HDBLUT, 'hdbv': HDBVLUT, 'hdbl': HDBLLUT, 'hdbt': HDBTLUT}

        # msb
        msb_lut = unit_dict[msb]
        # MSB
        self.msb_lut = msb_lut(*msb_weights, 2**self.msb_bits, upscale=upscale)
        #print("msb_lut",len(msb_weights))


        # LSB
        lsb_lut = unit_dict[lsb]
        self.lsb_lut = lsb_lut(*lsb_weights, 2**self.lsb_bits, upscale=upscale)
        #self.cnt=0


    def forward(self, img_lr):

        img_lr_255 = torch.floor(img_lr*255)
        img_lr_msb, img_lr_lsb = bit_plane_slicing(img_lr_255, self.bit_mask)
        #path_m='C:/Users/user/Simbury/study/fpga/hklut-main/testsets/test_data/SIDD_validation/msb'
        #path_l = 'C:/Users/user/Simbury/study/fpga/hklut-main/testsets/test_data/SIDD_validation/lsb'

        # msb
        img_lr_msb = torch.floor_divide(img_lr_msb, self.msb_step)
        #save_image(img_lr_msb, path_m + str(self.cnt) + '.png')
        MSB_out = self.msb_lut(img_lr_msb)/255.


        # lsb
        img_lr_lsb = torch.floor_divide(img_lr_lsb, self.lsb_step)
        #save_image(img_lr_lsb, path_l + str(self.cnt) + '.png')
        #self.cnt += 1

        LSB_out = self.lsb_lut(img_lr_lsb)/255.

        img_out = MSB_out + LSB_out + nn.Upsample(scale_factor=self.upscale, mode='nearest')(img_lr)
        
        return torch.clamp(img_out, 0, 1)

