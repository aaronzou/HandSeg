import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append('./dataset')
sys.path.append('./model')

from FCNet import VGGNet, FCN16s, SModel
import config

def show(direction='front'):
    if direction=='ego':
        root = "./ego_example"
    elif direction=='front':
        root = "./front_example"
    examples = [os.path.join(root, f) for f in os.listdir(root)]
    examples.sort()
    print("test {} examples on {}".format(str(len(examples)), direction))

    vgg_net = VGGNet(pretrained=True)

    if direction=='ego':
        model = FCN16s(pretrained_net=vgg_net, n_class=3)
        model.load_state_dict(torch.load(config.CHECKPOINT_EGO))
    elif direction=='front':
        model = SModel(pretrained_net=vgg_net, n_class=3)
        model.load_state_dict(torch.load(config.CHECKPOINT_FRONT))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    for item in examples:
        plt.subplot(1, 2, 1)
        plt.axis('off')
        tmp_depth_im = np.array(Image.open(item)).astype(np.float32)
        plt.imshow(tmp_depth_im)
        tmp_depth = torch.from_numpy(tmp_depth_im).type(torch.float32).unsqueeze(dim=2).expand(480, 640, 3)
        tmp_depth = tmp_depth.permute(2, 0, 1).unsqueeze(dim=0)

        predict = model(tmp_depth.to(device))
        pre_mask = torch.argmax(predict.cpu(), dim=1)[0].numpy()
        plt.subplot(1,2,2)
        plt.axis('off')
        plt.title('predict label')
        plt.imshow(pre_mask)
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', default='front', choices=['front', 'ego'], type=str)
    args = parser.parse_args()

    show(direction=args.direction)
