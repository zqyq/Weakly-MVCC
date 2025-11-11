import os
import re

import PIL.Image
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from torchvision.datasets import VisionDataset
# from camera_proj_Zhang import World2Image


class Pets(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # WILDTRACK has ij-indexing: H*W=480*1440, so x should be \in [0,480), y \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation,
        self.__name__ = 'Citystreet'
        self.img_shape = [768, 576]  # H,W; N_row,N_col
        # self.cam_range = [1, 3, 4]
        self.num_frame = 1500
        self.num_cam = 3
        self.grid_reduce = 1
        # dm
        self.unit = 10
        self.root = root
        self.train_view1_frame, self.train_view2_frame, self.train_view3_frame, self.train_GP = self.get_train_h5path()
        self.test_view1_frame, self.test_view2_frame, self.test_view3_frame, self.test_GP = self.get_test_h5path()
        # x,y actually means i,j in CityStreet, which correspond to h,w
        # self.indexing = 'xy'
        # #  for world map indexing

    # 注意frame_range belongs to Training: frame_0636.jpg---frame_1234.jpg, 300 images in total.
    # Testing: frame_1236.jpg---frame_1634.jpg, 200 images in total.
    # def

    def get_train_h5path(self):
        h5file_train_view1 = ['S1L3/14_17/train_test/PETS_S1L3_1_view1_train_test_10.h5', 'S1L3/14_33/train_test/PETS_S1L3_2_view1_train_test_10.h5',
                              'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view1_train_test_10.h5', 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view1_train_test_10.h5']
        h5file_train_view2 = ['S1L3/14_17/train_test/PETS_S1L3_1_view2_train_test_10.h5', 'S1L3/14_33/train_test/PETS_S1L3_2_view2_train_test_10.h5',
                              'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view2_train_test_10.h5', 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view2_train_test_10.h5']
        h5file_train_view3 = ['S1L3/14_17/train_test/PETS_S1L3_1_view3_train_test_10.h5', 'S1L3/14_33/train_test/PETS_S1L3_2_view3_train_test_10.h5',
                              'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view3_train_test_10.h5', 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view3_train_test_10.h5']
        h5file_train_GP = ['S1L3/14_17/GP_maps/PETS_S1L3_1_groundplane_dmaps_10.h5', 'S1L3/14_33/GP_maps/PETS_S1L3_2_groundplane_dmaps_10.h5',
                           'S2L2/14_55/GP_maps/PETS_S2L2_1_groundplane_dmaps_10.h5', 'S2L3/14_41/GP_maps/PETS_S2L3_1_groundplane_dmaps_10.h5']

        return h5file_train_view1, h5file_train_view2, h5file_train_view3, h5file_train_GP

    def get_test_h5path(self):
        h5file_test_GP = ['S1L1/13_57/GP_maps/PETS_S1L1_1_groundplane_dmaps_10.h5', 'S1L1/13_59/GP_maps/PETS_S1L1_2_groundplane_dmaps_10.h5',
                          'S1L2/14_06/GP_maps/PETS_S1L2_1_groundplane_dmaps_10.h5', 'S1L2/14_31/GP_maps/PETS_S1L2_2_groundplane_dmaps_10.h5']
        h5file_test_view1 = ['S1L1/13_57/train_test/PETS_S1L1_1_view1_train_test_10.h5', 'S1L1/13_59/train_test/PETS_S1L1_2_view1_train_test_10.h5',
                             'S1L2/14_06/train_test/PETS_S1L2_1_view1_train_test_10.h5', 'S1L2/14_31/train_test/PETS_S1L2_2_view1_train_test_10.h5']
        h5file_test_view2 = ['S1L1/13_57/train_test/PETS_S1L1_1_view2_train_test_10.h5', 'S1L1/13_59/train_test/PETS_S1L1_2_view2_train_test_10.h5',
                             'S1L2/14_06/train_test/PETS_S1L2_1_view2_train_test_10.h5', 'S1L2/14_31/train_test/PETS_S1L2_2_view2_train_test_10.h5']
        h5file_test_view3 = ['S1L1/13_57/train_test/PETS_S1L1_1_view3_train_test_10.h5', 'S1L1/13_59/train_test/PETS_S1L1_2_view3_train_test_10.h5',
                             'S1L2/14_06/train_test/PETS_S1L2_1_view3_train_test_10.h5', 'S1L2/14_31/train_test/PETS_S1L2_2_view3_train_test_10.h5']

        return h5file_test_view1, h5file_test_view2, h5file_test_view3, h5file_test_GP

    def get_img_fpath(self):
        viewimg_fpaths = {view: {} for view in range(1, 4)}
        for view, camera_folder in enumerate(sorted(os.listdir(os.path.join(self.root, 'image_frames')))):
            for fname in sorted(os.listdir(os.path.join(self.root, 'image_frames', camera_folder))):
                frame = int(fname.split('_')[1].split('.')[0])
                if frame in range(636, 1636, 2):
                    viewimg_fpaths[view + 1][frame] = os.path.join(self.root, 'image_frames', camera_folder, fname)
        return viewimg_fpaths


def test():
    from torchvision.transforms import ToTensor
    dataset = Citystreet(os.path.expanduser('~/Data/CityStreet'))
    # frame_range = range(636, 1236, 2)
    imgs_fpaths = dataset.get_img_fpath()
    # print(imgs_fpaths[1][636])
    # x = PIL.Image.open(imgs_fpaths[1][636]).convert('RGB')
    transform = ToTensor()
    # x = transform(x)
    data_sum, data_squared_sum = 0, 0
    for view in range(1, 4):
        for frame in range(636, 1236, 2):
            img = PIL.Image.open(imgs_fpaths[view][frame]).convert('RGB')
            img = transform(img)
            data_sum += torch.mean(img, dim=[1, 2])
            data_squared_sum += torch.mean(img ** 2, dim=[1, 2])
    city_mean = data_sum / 900
    city_std = (data_squared_sum / 900 - city_mean ** 2) ** 0.5
    print(city_mean, city_std)


if __name__ == '__main__':
    test()
