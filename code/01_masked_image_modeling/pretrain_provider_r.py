from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import random
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.augmentation import SimpleAugment as Filp

class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.simple_aug = Filp()
        self.crop_from_origin = [128, 128, 128]
        self.dataset = []
        self.att = []

        self.folder_name = os.path.join(cfg.DATA.data_folder)
        for subset in os.listdir(self.folder_name):
            if not os.path.isdir(os.path.join(self.folder_name, subset)):
                continue
            for file in os.listdir(self.folder_name + '/' + subset):
                if file.lower().endswith(('.tif', '.tiff')):
                    self.dataset.append(self.folder_name + '/' + subset + '/' + file)
        if not self.dataset:
            raise ValueError('No TIFF training stacks found in dataset subdirectories')

    def __getitem__(self, index):

        k = random.randint(0, len(self.dataset) - 1)
        used_data = imageio.volread(self.dataset[k])[:,12:-12,12:-12]
        raw_data_shape = used_data.shape

        random_z = random.randint(0, raw_data_shape[0]-self.crop_from_origin[0])
        random_y = random.randint(0, raw_data_shape[1]-self.crop_from_origin[1])
        random_x = random.randint(0, raw_data_shape[2]-self.crop_from_origin[2])

        imgs = used_data[random_z:random_z + self.crop_from_origin[0], \
                    random_y:random_y + self.crop_from_origin[1], \
                    random_x:random_x + self.crop_from_origin[2]].copy()


        [imgs] = self.simple_aug([imgs])

        imgs = self.scaler(imgs)
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        return imgs

    def __len__(self):
        return int(sys.maxsize)

    def scaler(self, img):
        return np.float32(img) / 255.0


class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
            return batch
