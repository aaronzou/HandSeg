from __future__ import division

import sys
sys.path.append('../../')

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import config


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)
    size.append(N)
    return ones.view(*size)



class HandSegDataset(Dataset):
    def __init__(self, direction='ego', is_train=True):
        super(HandSegDataset, self).__init__()

        if direction == 'ego':
            if is_train:
                self.data = np.load(config.TMP_HAND_SEG_EGO_DATA)
            else:
                self.data = np.load(config.TMP_HAND_SEG_EGO_TEST_DATA)
        elif direction == 'front':
            if is_train:
                self.data = np.load(config.TMP_HAND_SEG_FRONT_DATA)
            else:
                self.data = np.load(config.TMP_HAND_SEG_FRONT_TEST_DATA)

        self.rgb_dirs = self.data['rgb']
        self.depth_dirs = self.data['depth']
        self.tmp_depth_dirs = self.data['tmp_depth']
        self.mask_dirs = self.data['mask']

    def __len__(self):
        if len(self.depth_dirs) != len(self.mask_dirs):
            assert "data wrong!"
        return len(self.depth_dirs)


    def __getitem__(self, index):
        item = {}

        rgb_dir = self.rgb_dirs[index]
        depth_dir = self.depth_dirs[index]
        tmp_depth_dir = self.tmp_depth_dirs[index]
        mask_dir = self.mask_dirs[index]

        rgb_im = np.array(Image.open(rgb_dir)).astype(np.float32)
        depth_im = np.array(Image.open(depth_dir)).astype(np.float32)

        tmp_depth_im = np.array(Image.open(tmp_depth_dir)).astype(np.float32)
        tmp_depth = torch.from_numpy(tmp_depth_im).type(torch.float32).unsqueeze(dim=2).expand(480, 640, 3)
        tmp_depth = tmp_depth.permute(2, 0, 1)

        mask_im = np.array(Image.open(mask_dir)).astype(np.float32)
        # background:0, left:2, right:1
        for i in range(480):
            for j in range(640):
                if mask_im[i, j] == 255:
                    mask_im[i, j] = 0
                elif mask_im[i, j] == 0:
                    mask_im[i, j] = 2
                elif mask_im[i, j] == 1:
                    mask_im[i, j] = 1

        mask_onehot = get_one_hot(torch.LongTensor(mask_im), 3)
        mask_onehot = mask_onehot.permute(2, 0, 1).type(torch.float32)

        item['rgb_dir'] = rgb_dir
        item['rgb'] = rgb_im
        item['depth_dir'] = depth_dir
        item['depth_im'] = depth_im
        item['tmp_depth_dir'] = tmp_depth_dir
        item['tmp_depth'] = tmp_depth
        item['mask_dir'] = mask_dir
        item['mask_im'] = torch.from_numpy(mask_im).type(torch.float32)
        item['mask_onehot'] = mask_onehot
        return item



if __name__ == '__main__':
    data = HandSegDataset(direction='front', is_train=False)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    for item in data_loader:
        rgb_dir = item['rgb_dir']
        depth_dir = item['depth_dir']
        mask_dir = item['mask_dir']
        depth_im = item['depth_im']
        mask_im = item['mask_im']
        mask_onehot = item['mask_onehot']


        mask_im = torch.squeeze(mask_im, dim=0).numpy()
        mask_im = mask_im.astype(np.float32)
        print(mask_onehot.shape, mask_im.shape, item['mask_im'].shape)
        mask = mask_onehot[0].permute(1, 2, 0)
        mask = mask.numpy()
        mask = np.argmax(mask, -1)
        print((mask == mask_im).all())


        plt.subplot(1, 3, 1)
        plt.axis('off')
        rgb = item['rgb']
        rgb = torch.squeeze(rgb).numpy()
        plt.imshow(rgb/255)


        plt.subplot(1, 3, 2)
        plt.axis('off')
        depth_im = torch.squeeze(depth_im).numpy()
        plt.imshow(depth_im)


        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(mask)
        plt.show()

        tmp_depth = item['tmp_depth']
        tmp_depth = torch.squeeze(tmp_depth).numpy()
        plt.imshow(tmp_depth[0])
        plt.show()


        break