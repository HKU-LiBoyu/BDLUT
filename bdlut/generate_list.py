import os

PATH = '/home/richard/datasets/GOPRO_LARGE'
#PATH_train = '/home/richard/datasets/GOPRO_LARGE/train'
#PATH_val = '/home/richard/datasets/GOPRO_LARGE/test'
folder_train_list=os.listdir(os.path.join(PATH,'train'))
folder_val_list=os.listdir(os.path.join(PATH,'test'))
img_list_f=[]
for fd_t in folder_train_list:
    PATH_B = os.path.join('train', fd_t, 'blur')
    img_list = os.listdir(os.path.join(PATH,'train', fd_t, 'blur'))
    img_list_f = img_list_f+[os.path.join(PATH_B, f_tb) for f_tb in img_list]
with open('GOPRO_Train_blur.txt','w') as f:
    for i in img_list_f:
        f.write(i+'\n')

img_list_fs=[]
for fd_t in folder_train_list:
    PATH_S = os.path.join('train', fd_t, 'sharp')
    img_list = os.listdir(os.path.join(PATH,'train', fd_t, 'sharp'))
    img_list_fs = img_list_fs+[os.path.join(PATH_S, f_tb) for f_tb in img_list]

with open('GOPRO_Train_sharp.txt','w') as f:
    for i in img_list_fs:
        f.write(i + '\n')

img_list_fv=[]
for fd_v in folder_val_list:
    PATH_B = os.path.join('test', fd_v, 'blur')
    img_list = os.listdir(os.path.join(PATH,'test', fd_v, 'blur'))
    img_list_fv = img_list_fv+[os.path.join(PATH_B, f_vb) for f_vb in img_list]

with open('GOPRO_Val_blur.txt','w') as f:
    for i in img_list_fv:
        f.write(i + '\n')

img_list_fvs=[]
for fd_v in folder_val_list:
    PATH_S = os.path.join('test', fd_v, 'sharp')
    img_list = os.listdir(os.path.join(PATH,'test', fd_v, 'sharp'))
    img_list_fvs = img_list_fvs+[os.path.join(PATH_S, f_vs) for f_vs in img_list]

with open('GOPRO_Val_sharp.txt','w') as f:
    for i in img_list_fvs:
        f.write(i + '\n')