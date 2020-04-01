import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_gif(root='./results_2', name='res2.gif', modify=False, duration=0.05):
    image_paths = [os.path.join(root, f)
                   for f in os.listdir(root)]
    image_paths.sort()
    print("{} images.".format(str(len(image_paths))))

    frames = []
    for i in range(746,857):#271
        image = imageio.imread(image_paths[i])
        # image = np.array(Image.open(image_paths[i])).astype(np.float32)
        # image = np.array(plt.imread(image_paths[i]))
        if modify:
            for i in range(480):
                for j in range(640):
                    if image[i, j] == 255:
                        image[i, j] = 0
                    elif image[i, j] == 0:
                        image[i, j] = 2
                    elif image[i, j] == 1:
                        image[i, j] = 1
        # image = Image.fromarray(image)
        frames.append(image)
    imageio.mimsave(name, frames, 'GIF', duration=duration)

def gt_label(name='gt1.gif', duration=0.2):
    from seg_dataset import HandSegDataset
    import torch
    data = HandSegDataset(is_train=False)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    frames = []
    for step, item in enumerate(data_loader, 0):
        mask_im = torch.squeeze(item['mask_im'][0], dim=0).numpy()
        mask_im = mask_im.astype(np.float32)
        frames.append(mask_im)
        print(step)
        if step == 20:
            break

    imageio.mimsave(name, frames, 'GIF', duration=duration)
    print("done")

if __name__ == '__main__':
    label_path= '/home/liwensh2/code/HandsTrack/dataset/paintedHands/ego/output_user04/label_filtered'
    get_gif(name='gif_res/res13.gif', root='results_front_2', modify=False)
    # get_gif(root='./results', name='res1.gif', modify=False)
    # gt_label()
