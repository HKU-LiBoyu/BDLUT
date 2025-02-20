import os
import random
import sys

import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import modcrop, rgb2ycbcr



class Provider(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size, qf):
        self.data = DIV2K(scale, path, patch_size, qf=qf)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True,qf=20):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.qf = qf # quality factor
        fl = os.listdir(os.path.join(path, "DIV2K_data"))
        self.file_list = [f[:-4] for f in fl] 
        print("deblocking qf is ",self.qf)

        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

        self.lr_cache = os.path.join(path, "cache_lr_{}.npy".format(self.qf))
        if not os.path.exists(self.lr_cache):
            self.cache_lr()
            print("LR image cache to:", self.lr_cache)
        self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
        print("LR image cache from:", self.lr_cache)

    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "DIV2K_data")
        for f in self.file_list:
            hr = rgb2ycbcr(np.array(Image.open(os.path.join(dataLR, f+".png"))))
            _, encimg = cv2.imencode('.jpg', hr, [int(cv2.IMWRITE_JPEG_QUALITY), self.qf])
            lr_dict[f] = cv2.imdecode(encimg, 0)
        np.save(self.lr_cache, lr_dict, allow_pickle=True)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "DIV2K_data")
        for f in self.file_list:
            hr_dict[f] = rgb2ycbcr(np.array(Image.open(os.path.join(dataHR, f+".png"))))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]
        im = self.lr_ims[key]

        shape = lb.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        # c = random.choice([0, 1, 2])

        lb = lb[i:i+self.sz, j:j+self.sz]
        im = im[i:i+self.sz, j:j+self.sz]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.expand_dims(lb.astype(np.float32)/255.0, axis=0)
        im = np.expand_dims(im.astype(np.float32)/255.0, axis=0)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class SRBenchmark(Dataset):
    def __init__(self, path, scale=4, qf=20):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = (5 + 29) * 2

        print("The benchmark dataset deblocking quality factor is: ", qf)

        for dataset in ['classic5', 'LIVE1']:
            folder = os.path.join(path, dataset)
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, files[i])))
                #im_hr = modcrop(im_hr, scale)
                if dataset == 'classic5': # gray
                    pass
                    # im_hr = im_hr[:, :, np.newaxis]
                elif dataset == 'LIVE1': # color
                    im_hr = rgb2ycbcr(im_hr)

                key = dataset + '_' + files[i][:-4] # remove .png e.g. Set5_baby
                self.ims[key] = im_hr[:, :, np.newaxis] # no norm

                _, encimg = cv2.imencode('.jpg', im_hr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
                im_lr = cv2.imdecode(encimg, 0)


                key = dataset + '_' + files[i][:-4] + 'n%d' % qf # e.g. Set5_babyn15
                self.ims[key] = im_lr.astype(np.float32)[:, :, np.newaxis] / 255.0

                assert (len(im_lr.shape) == len(im_hr.shape) == 2)

        assert (len(self.ims.keys()) == _ims_all)
