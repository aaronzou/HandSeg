from __future__ import division

import sys
sys.path.append('../dataset/')
sys.path.append('../model/')
sys.path.append('../')

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seg_dataset import HandSegDataset
from FCNet import VGGNet, FCN16s
from loss_function import cross_entropy2d


def trainer(device, batch_size=4, eqoch_num=20):
    seg_data = HandSegDataset(is_train=True)
    data_loader = DataLoader(seg_data, batch_size=batch_size, num_workers=4, shuffle=True)

    vgg_model = VGGNet(requires_grad=True)
    model = FCN16s(pretrained_net=vgg_model, n_class=3)
    model = model.to(device)

    # criterion = cross_entropy2d().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    all_train_iter_loss = []

    writer = SummaryWriter("./checkpoints/tmp_FNC16s/")
    tol_step = 0

    for epoch in range(eqoch_num):
        train_loss = 0.0
        model.train()

        for step, batch in enumerate(tqdm(data_loader, desc='Epoch ' + str(epoch),
                                          total=len(seg_data) // batch_size,
                                          initial=0),0):
            item = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            depth_im = item['tmp_depth'].to(device)
            mask_onehot = item['mask_onehot'].to(device)
            mask_im = item['mask_im'].to(device)
            optimizer.zero_grad()
            output = model(depth_im).type(torch.float32)
            output = torch.sigmoid(output)
            loss = cross_entropy2d(output, mask_im.to(torch.long))
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()
            tol_step = tol_step+1
            writer.add_scalar('loss', iter_loss * 500, tol_step)

        torch.save(model, './checkpoints/tmp_FCN16s_{}.pt'.format(epoch))
    writer.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    trainer(device=device)

