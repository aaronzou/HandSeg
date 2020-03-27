import numpy as np
import os
import sys
sys.path.append('./')
import config

def GenerateData():
    root = config.HAND_SEG_EGO_DATA_PATH

    train_user_dir = ['output_user01', 'output_user02']
    test_user_dir = ['output_user04']

    train_depth, train_tmp_depth, train_mask, train_rgb, \
    test_depth, test_tmp_depth, test_mask, test_rgb = [], [], [], [],[], [], [], []
    for i in range(len(train_user_dir)):
        path = os.path.join(root, train_user_dir[i])
        depth = [os.path.join(path, 'depth', f)
                 for f in os.listdir(os.path.join(path, 'depth'))]
        depth.sort()
        tmp = [os.path.join(path, 'tmp_depth', f)
                 for f in os.listdir(os.path.join(path, 'tmp_depth'))]
        tmp.sort()
        mask = [os.path.join(path, 'label_filtered', f)
                for f in os.listdir(os.path.join(path, 'label_filtered'))]
        mask.sort(key = lambda x: str(x[:-4]))
        rgb = [os.path.join(path, 'color', f)
                for f in os.listdir(os.path.join(path, 'color'))]
        rgb.sort()
        print(len(rgb), len(depth), len(tmp), len(mask))
        for t in range(len(tmp)):
            train_depth.append(depth[t])
            train_tmp_depth.append(tmp[t])
            train_mask.append(mask[t])
            train_rgb.append(rgb[t])

    for i in range(len(test_user_dir)):
        path = os.path.join(root, test_user_dir[i])
        depth = [os.path.join(path, 'depth', f)
                 for f in os.listdir(os.path.join(path, 'depth'))]
        depth.sort()
        tmp = [os.path.join(path, 'tmp_depth', f)
                 for f in os.listdir(os.path.join(path, 'tmp_depth'))]
        tmp.sort()
        mask = [os.path.join(path, 'label_filtered', f)
                for f in os.listdir(os.path.join(path, 'label_filtered'))]
        mask.sort(key = lambda x: str(x[:-4]))
        rgb = [os.path.join(path, 'color', f)
                for f in os.listdir(os.path.join(path, 'color'))]
        rgb.sort()
        print(len(rgb), len(depth), len(tmp), len(mask))
        for t in range(len(tmp)):
            test_depth.append(depth[t])
            test_tmp_depth.append(tmp[t])
            test_mask.append(mask[t])
            test_rgb.append(rgb[t])

    print(len(train_depth), len(train_mask))
    print(len(test_depth), len(test_tmp_depth))
    np.savez('ego_train.npz', rgb=train_rgb, depth=train_depth, tmp_depth=train_tmp_depth, mask=train_mask)
    np.savez('ego_test.npz', rgb=test_rgb, depth=test_depth, tmp_depth=test_tmp_depth,mask=test_mask)



if __name__ == '__main__':
    GenerateData()