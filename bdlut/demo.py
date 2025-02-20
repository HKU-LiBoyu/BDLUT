from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torchvision


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_list[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.img_list[idx]
    

# 设置图像的预处理转换
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 调整图像大小为256x256像素
    transforms.ToTensor()  # 将图像转换为张量
])

# 数据集的根目录
data_dir = '/home/boyu/hklut-main/dataset/128p'
os.makedirs(f'{data_dir}/512p', exist_ok=True)

# 使用ImageFolder类加载数据集
dataset = CustomDataset(data_dir=data_dir, transform=transform)
#%%
# 创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#%%


LUT_dir = './luts'
msb = 'hdbl'
lsb = 'hd'
act_fn = 'relu'
n_filters = 64
upscale = [2, 2]
factors = 'x'.join([str(s) for s in upscale])

exp_name = "msb_{}_lsb_{}_act_{}_nf_{}_{}".format(msb, lsb, act_fn, n_filters, factors)
lut_path = f'{LUT_dir}/{exp_name}'

#%%
import numpy as np
from models import HKLUT

device = 'cuda'
luts = []
n_stages = 1
sr_scale = 4
models = []




# Load LUTs
lut_files = os.listdir(lut_path)
print("LUT path: ", lut_path)
for stage in range(n_stages):
    # msb
    msb_weights = []
    for ktype in msb:
        weight = torch.tensor(np.load(os.path.join(lut_path, f'S{stage}_MSB_{msb.upper()}_LUT_{ktype.upper()}_x{upscale[stage]}_4bit_int8.npy')).astype(np.int_))
        msb_weights.append(weight)

    # lsb
    lsb_weights = []
    for ktype in lsb:
        weight = torch.tensor(np.load(os.path.join(lut_path, f'S{stage}_LSB_{lsb.upper()}_LUT_{ktype.upper()}_x{upscale[stage]}_4bit_int8.npy')).astype(np.int_))
        lsb_weights.append(weight)

    models.append(HKLUT(msb_weights, lsb_weights, msb=msb, lsb=lsb, upscale=upscale[stage]).to(device))

#%%
with torch.no_grad():
    for model in models:
        model.eval()

    for i, (x, path) in enumerate((dataloader)):
        for model_idx, model in enumerate(models):
#            x, MSB, LSB = model(x)
            # x, MSB, LSB = model(x)
#             print("x shape:", x.shape)
#             print("x values:", x)

            # 计算img_lr_255并打印其尺寸和具体值
#             img_lr_255 = torch.floor(x * 255)
#             print("img_lr_255 shape:", img_lr_255.shape)
#             print("img_lr_255 values:", img_lr_255)

#             if model_idx == 0:
           # 保存x到文本文件
    #            print(model_idx)
#               with open(f'output_x_{model_idx}.txt', 'w') as f:
#                 x_np = x.cpu().numpy()
#                 for idx in range(x_np.shape[0]):
#                   for j in range(x_np.shape[1]):
#                    for k in range(x_np.shape[2]):
#                         pixel = x_np[idx, j, k]
#                         f.write(' '.join([str(val) for val in pixel]) + '\n')
 
           # 保存img_lr_255到文本文件
#               with open(f'output_img_lr_255_{model_idx}.txt', 'w') as f:
#                 img_lr_255_np = img_lr_255.cpu().numpy()
#                 for idx in range(img_lr_255_np.shape[0]):
#                   for j in range(img_lr_255_np.shape[1]):
#                     for k in range(img_lr_255_np.shape[2]):
#                         pixel = img_lr_255_np[idx, j, k]
#                         f.write(' '.join([str(val) for val in pixel]) + '\n')
             x, MSB, LSB = model(x)
             np.save(f"./MSB_S{model_idx}.npy", MSB)
             np.save(f"./LSB_S{model_idx}.npy", LSB)
                     # 打印x的尺寸和具体值

        # Output 
        image_out = (x).cpu().data.numpy()
        image_out = np.floor(image_out*255) # (1, 3, 512, 512)
        # image_out = np.round(image_out*255) # (1, 3, 512, 512)
        image_out = np.transpose(np.clip(image_out[0], 0, 255), [1,2,0]) # BxCxHxW -> HxWxC

        # Save to file
        image_out = image_out.astype(np.uint8)
        Image.fromarray(image_out).save(f'{data_dir}/512p/{path[0][:-4]}.png')
        # break
    