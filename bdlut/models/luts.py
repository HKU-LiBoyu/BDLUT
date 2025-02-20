import torch
import torch.nn as nn
import torch.nn.functional as F

class HDVLUT(nn.Module):
    def __init__(self, h_weight, d_weight, v_weight, L, upscale=2):
        super(HDVLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.v_weight = v_weight
        self.rot_dict = {'h': [0, 2], 'd': [0, 1, 2, 3], 'v': [0, 2]}
        self.pad_dict = {'h': (0,1,0,0), 'd': (0,1,0,1), 'v': (0,0,0,1)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'v']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 0:0+H, 1:1+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 1:1+H, 1:1+W]
                else: # v
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 1:1+H, 0:0+W]
                    weight = self.v_weight

                tmp = weight[img_a.flatten()*self.L + img_b.flatten()].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute(0, 1, 2, 4, 3, 5).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


class HLLUT(nn.Module):
    def __init__(self, h_weight, l_weight, L, upscale=2):
        super(HLLUT, self).__init__()
        self.h_weight = h_weight
        self.l_weight = l_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'l': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 'l': (0, 2, 0, 2)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'l']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 'l':
                    weight = self.l_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 1:1+H, 1:1+W]

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


class HSLUT(nn.Module):
    def __init__(self, h_weight, s_weight, L, upscale=2):
        super(HSLUT, self).__init__()
        self.h_weight = h_weight
        self.s_weight = s_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 's': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 's': (0, 2, 0, 2)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 's']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 's':
                    weight = self.s_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor
    
    
    
class HDBLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, L, upscale=2):
        super(HDBLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2)}
        self.avg_factor = 3.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]
                else:
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 2:2+W]
                    img_c = img_in[:, :, 2:2+H, 1:1+W]
                    weight = self.b_weight

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


# To make it to be an exponential multiple of 2 -> shift operation
class HDLUT(nn.Module):
    def __init__(self, h_weight, d_weight, L, upscale=2):
        super(HDLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 1, 0, 0), 'd': (0, 1, 0, 1)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                else:
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 1:1 + W]


                tmp = weight[img_a.flatten() * self.L + img_b.flatten()].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2, 3])

        return out / self.avg_factor


class HDBVLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, v_weight, L, upscale=2):
        super(HDBVLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.v_weight = v_weight
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 2, 0, 2)
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'v']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                    img_c = img_in[:, :, 0:0 + H, 2:2 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 1:1 + W]
                    img_c = img_in[:, :, 2:2 + H, 2:2 + W]
                elif ktype == 'b':
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 2:2 + W]
                    img_c = img_in[:, :, 2:2 + H, 1:1 + W]

                elif ktype == 'v':
                    weight = self.v_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 0:0 + W]
                    img_c = img_in[:, :, 2:2 + H, 0:0 + W]


                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                tmp = torch.rot90(tmp, 4 - r, [2, 3])
                out += tmp

        return out / self.avg_factor
    
    
class HDBLLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, l_weight, L, upscale=2):
        super(HDBLLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.l_weight = l_weight
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 2, 0, 2)
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'l']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                    img_c = img_in[:, :, 0:0 + H, 2:2 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 1:1 + W]
                    img_c = img_in[:, :, 2:2 + H, 2:2 + W]
                elif ktype == 'b':
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 2:2 + W]
                    img_c = img_in[:, :, 2:2 + H, 1:1 + W]

                elif ktype == 'l':
                    weight = self.l_weight#v_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                    img_c = img_in[:, :, 1:1 + H, 1:1 + W]


                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                tmp = torch.rot90(tmp, 4 - r, [2, 3])
                out += tmp

        return out / self.avg_factor

class HDBTLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, t_weight, L, upscale=2):
        super(HDBTLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.t_weight = t_weight
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 2, 0, 2)
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 't']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                    img_c = img_in[:, :, 0:0 + H, 2:2 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 1:1 + W]
                    img_c = img_in[:, :, 2:2 + H, 2:2 + W]
                elif ktype == 'b':
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 2:2 + W]
                    img_c = img_in[:, :, 2:2 + H, 1:1 + W]

                elif ktype == 't':
                    weight = self.t_weight#v_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 0:0 + W]
                    img_c = img_in[:, :, 0:0 + H, 1:1 + W]


                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                tmp = torch.rot90(tmp, 4 - r, [2, 3])
                out += tmp

        return out / self.avg_factor