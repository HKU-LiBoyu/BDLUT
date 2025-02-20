import torch
import torch.nn as nn
import torch.nn.functional as F


class Decompose(nn.Module):
    def __init__(self, size=5, dim=64):
        super().__init__()
        self.size = size
        self.dim = dim
        for i in range(size*size):
            setattr(self, f'conv1x1_{i}_E', nn.Conv2d(1, dim, 1))
            setattr(self, f'conv1x1_{i}_S', nn.Conv2d(dim, 1, 1))
            
        
    def forward(self, x):
        # input: (batch_size, channel, H, W)
        bs, C, H, W = x.shape
        x = torch.clamp(x, 0.0, 1.0)
        x = x.view(bs*C, 1, H, W)
        output = torch.zeros_like(x)
        
        padding = self.size // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        for i in range(self.size):
            for j in range(self.size):
                x_i = x_pad[:, :, i:i+H, j:j+W]
                x_i_E = getattr(self, f'conv1x1_{i*self.size+j}_E')(x_i)
                x_i_S = getattr(self, f'conv1x1_{i*self.size+j}_S')(x_i_E)
                # output += x_i_S.int()
                output += x_i_S
    
        output /= (self.size*self.size)
        # output = output.int()
        output = torch.clamp(output, 0.0, 1.0)
        
        return output.view(bs, C, H, W)
    
    
if __name__ == '__main__':
    model = Decompose(size=7, dim=32)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)