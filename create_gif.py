import os
import imageio
import numpy as np

def get_gif(root='./results', name='res.gif', duration=0.2):
    image_paths = [os.path.join(root, f)
                   for f in os.listdir(root)]
    image_paths.sort()
    print("{} images.".format(str(len(image_paths))))

    frames = []
    for path in image_paths:
        depth_im = imageio.imread(path)
        frames.append(depth_im)

    imageio.mimsave(name, frames, 'GIF', duration=duration)
    print("done")

if __name__ == '__main__':
    label_path= '/home/liwensh2/code/HandsTrack/dataset/paintedHands/ego/output_user04/label_filtered'
    get_gif(name='res.gif', root='./results')