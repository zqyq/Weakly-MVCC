import os
import json

import scipy
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *
from multiview_detector.loss.projasa import *
import matplotlib.pyplot as plt
import os
# import sys
import numpy as np
from lb_utils import _transform
# import h5py
# from scipy import ndimage
# import scipy
# import scipy.ndimage
# import scipy.io as sio
import torch.nn.functional as F

# from sklearn import feature_extraction
import json
import cv2
import random

from datetime import datetime
import collections


class frameDataset(VisionDataset):
    def __init__(self, base, dataset_dir=None, batch_size=1, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True, seed=14, isTest=False):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.train = train
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        # CVCS dataset parameters:
        data_file = '/mnt/d/data/CVCS_dataset/'
        label_file = '/mnt/d/data/CVCS_dataset/labels/100frames_labels_reproduce_640_480_CVCS/'
        depth_file = '/mnt/d/data/CVCS_dataset/labels/100frames_depthMaps_reproduce_640_480/'

        # data_file = '/mnt/data/Datasets/LCVCS/'
        # label_file = '/mnt/data/Datasets/LCVCS/'
        # read images
        if train:
            self.file_path = data_file + 'train/'
            self.label_file_path = label_file + 'train/'
            self.depth_file_path = depth_file + 'train/'
        else:
            self.file_path = data_file + 'val/'
            self.label_file_path = label_file + 'val/'
            self.depth_file_path = depth_file + 'val/'

        self.gt_fpath = self.file_path

        ind_scene = 0
        nb_batch_used = 0
        nb_view_used = 0

        # wld_h = int(720/2) #480  #360
        # wld_w = int(640/2) #640

        if train:
            self.batch_size = 2  # batch_size
        else:
            self.batch_size = 1

        # self.cropped_size = [180, 160]
        # self.cropped_size = [300, 200]
        self.cropped_size = [160, 180]

        self.view_size = 5
        # self.view_size = 2000
        self.num_cam = self.view_size
        self.patch_num = 1 #5

        # r 为坐标缩放比
        self.r = 5
        # 2 # 0.5m/pixel
        # 10: 0.1m/pixel
        # 5: 0.2m/pixel

        # a,b 为图像边缘扩张, 防止在图像边缘的人被漏掉; a对应w, b对应h
        self.a = 5
        self.b = 5
        # cropped_size, r, a, b, patch_num
        # 28 Epoch and 100 Epoch seed is 14, train img
        # then train map
        # random.seed(14)
        # random.seed(15)
        random.seed(seed)

        # list M scenes:
        # self.scene_name_list = os.listdir(self.file_path)  # 2scenes model training
        # self.scene_name_list = self.scene_name_list[0:15]

        if train:
            self.scene_name_list = os.listdir(self.file_path)
            # self.scene_name_list = ['scene_80']
            self.remove_list = ['scene_05', 'scene_06', 'scene_12', 'scene_18', 'scene_02', 'scene_15', 'scene_44', 'scene_42']
            for scene in self.remove_list:
                self.scene_name_list.remove(scene)
            # self.nb_samplings = 5  # int(nb_frames/view_size/batch_size + 1)#*3 for epochFixed *3; for epochFixed2, not.
            self.nb_scenes = len(self.scene_name_list)
            self.nb_samplings = 5
        else:
            self.scene_name_list = os.listdir(self.file_path)
            # self.nb_samplings = 21  # int(nb_frames/view_size/batch_size + 1)
            self.nb_samplings = 21 if isTest else 1
            self.nb_scenes = len(self.scene_name_list)

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        # self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        # self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        # self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

        # only use head labels:
        self.img_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        # self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def read_json_frame(self, coords_info, cropped_size, r, a, b, patch_num):

        # set the wld map paras
        wld_map_paras = coords_info['wld_map_paras']
        s, r0, a0, b0, h4, w4, d_delta, d_mean, w_min, h_min = wld_map_paras  # old 640*480 labels

        # reconstruct wld_map_paras:
        # a = (w - (w_max - w_min) * r) / 2
        # b = (h - (h_max - h_min) * r) / 2
        w_max = (4 * w4 - 2 * a0) / r0 + w_min
        h_max = (4 * h4 - 2 * b0) / r0 + h_min

        # actual size:
        w_actual = int((w_max - w_min + 2 * a)) * r
        h_actual = int((h_max - h_min + 2 * b)) * r

        h_actual = int(max(h_actual, cropped_size[0] + patch_num))
        w_actual = int(max(w_actual, cropped_size[1] + patch_num))

        # create patch_num patches' bbox coordinates:
        ## h: 0~h_actual-cropped_size[0], w: 0~w_actual-cropped_size[1]
        h_range = range(0, h_actual - cropped_size[0] + 1)
        w_range = range(0, w_actual - cropped_size[1] + 1)

        h_random = random.sample(h_range, k=patch_num)
        # h_random = [h_actual]
        w_random = random.sample(w_range, k=patch_num)
        # w_random = [w_actual]
        hw_random = np.asarray([h_random, w_random])
        # self.cropped_size = [h_actual, w_actual]

        # h_random = np.linspace(0, h_actual-1, h_actual)
        # w_random = np.linspace(0, w_actual-1, h_actual)
        # h_coordinates, w_coordinates = np.meshgrid(h_random, w_random)

        wld_map_paras = [r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max]

        # get people 2D and 3D coords:
        coords = coords_info['image_info']

        # coords_2d_all = [] #np.zeros((1, 2))
        coords_3d_id_all = []  # np.zeros((1, 3))

        # id = 0
        for point in coords:
            id = point['idx']
            coords_3d_id = point['world'] + [id]
            coords_3d_id_all.append(coords_3d_id)

        coords_3d_id_all = np.asarray(coords_3d_id_all, dtype='float32')

        # form the para list:
        return coords_3d_id_all, wld_map_paras, hw_random

    def read_json_view(self, coords_info, flag=False):
        # get camera matrix
        cameraMatrix = np.asarray(coords_info['cameraMatrix'])
        fx = cameraMatrix[0][0]
        fy = cameraMatrix[1][1]
        u = cameraMatrix[0][2]
        v = cameraMatrix[1][2]

        # get camera matrix:
        distCoeffs = coords_info['distCoeffs']

        # 相机旋转矩阵
        rvec = coords_info['rvec']
        # 相机平移矩阵
        tvec = coords_info['tvec']

        camera_paras = [fx] + [fy] + [u] + [v] + distCoeffs + rvec + tvec
        camera_paras = np.asarray(camera_paras)

        # get people 2D and 3D coords:
        coords = coords_info['image_info']

        coords_2d_all = []  # np.zeros((1, 2))
        coords_3d_id_all = []  # np.zeros((1, 3))
        coords_3d_id_all_of_json = []

        for point in coords:
            id = point['idx']

            coords_2d0 = point['pixel']

            coords_3d_id = point['world'] + [id]
            coords_3d_id_all_of_json.append(coords_3d_id)

            if coords_2d0 == None:
                continue

            if flag:
                coords_2d = [coords_2d0[1] / 1920.0, coords_2d0[0] / 1080.0] + [id]
            else:
                coords_2d = [coords_2d0[1] / 1920.0, coords_2d0[0] / 1080.0]

            # coords_3d_id = point['world'] + [id]

            # coords_2d_all = np.concatenate((coords_2d_all, np.expand_dims(coords_2d, axis=0)), axis=0)
            # coords_3d_all = np.concatenate((coords_3d_all, np.expand_dims(coords_3d, axis=0)), axis=0)
            coords_2d_all.append(coords_2d)
            coords_3d_id_all.append(coords_3d_id)

        # form the para list:
        return coords_3d_id_all, coords_2d_all, camera_paras, coords_3d_id_all_of_json

    def density_map_creation(self, pmap, w, h):
        if pmap.size==0:
            img_pmap_i = np.zeros((h, w))
            density_map = np.asarray(img_pmap_i).astype('f')

        else:
            density_map = []
            img_id_all = pmap[:, 0].shape[0]

            pmap = np.asarray(pmap)
            img_dmap = np.zeros((h, w))

            x = (pmap[:, 0]*w).astype('int')
            y = (pmap[:, 1]*h).astype('int')
            img_dmap[y, x] = 1

            density_map = np.asarray(img_dmap).astype('f')

        return density_map


    def GP_density_map_creation(self, wld_coords, crop_size, wld_map_paras, hw_random):
        # we need to resize the images
        h = int(crop_size[0])
        w = int(crop_size[1])

        r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max = wld_map_paras

        h_actual, w_actual = int(h_actual), int(w_actual)
        patch_num = hw_random.shape[1]

        if wld_coords.size == 0:
            img_pmap_i = np.zeros((h_actual, w_actual))
            GP_density_map_0 = img_pmap_i

            GP_density_map = []
            for p in range(patch_num):
                hw = hw_random[:, p]
                GP_density_map_i = img_pmap_i[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
                GP_density_map.append(GP_density_map_i)

            GP_density_map = np.asarray(GP_density_map)

        else:

            wld_coords_transed = np.zeros(wld_coords.shape)
            wld_coords_transed[:, 0] = (wld_coords[:, 0] - w_min + a) * r
            wld_coords_transed[:, 1] = (wld_coords[:, 1] - h_min + b) * r
            wld_coords_transed = wld_coords_transed.astype('int')

            assert min(wld_coords_transed[:, 0]) >= 0 and max(wld_coords_transed[:, 0]) < w_actual
            assert min(wld_coords_transed[:, 1]) >= 0 and max(wld_coords_transed[:, 1]) < h_actual

            img_id_all = wld_coords_transed[:, 0].shape[0]

            # img_pmap = np.zeros((h_actual, w_actual, img_id_all))
            # img_pmap[wld_coords_transed[:, 1], wld_coords_transed[:, 0], range(img_id_all)] = 1

            img_pmap = np.zeros((h_actual, w_actual))
            img_pmap[wld_coords_transed[:, 1], wld_coords_transed[:, 0]] = 1

            # patch_num = hw_random.shape[1]
            sg = 2
            sigma = [sg, sg]
            sg_size = 4 * sg

            dmap_i = np.zeros((2 * sg_size + 1, 2 * sg_size + 1))
            dmap_i[sg_size, sg_size] = 1
            dmap_i = scipy.ndimage.gaussian_filter(dmap_i, sigma, mode='reflect')
            # print(sum(dmap_i.flatten()))

            img_dmap = []  # np.zeros((h, w, img_id_all))
            for i in range(img_id_all):
                img_dmap_i = np.zeros((h_actual + 2 * sg_size, w_actual + 2 * sg_size))
                img_dmap_i[wld_coords_transed[i, 1]:(wld_coords_transed[i, 1] + 2 * sg_size + 1),
                wld_coords_transed[i, 0]:(wld_coords_transed[i, 0] + 2 * sg_size + 1)] = dmap_i
                img_dmap.append(img_dmap_i[sg_size:h_actual + sg_size, sg_size:w_actual + sg_size])
            img_dmap = np.asarray(img_dmap)
            GP_density_map_0 = np.sum(img_dmap, axis=0).astype('f')  # -1

            # patch_num = hw_random.shape[1]
            # GP_density_map = []
            # for p in range(patch_num):
            #     hw = hw_random[:, p]
            #     GP_density_map_i = GP_density_map_0[hw[0]:hw[0] + h, hw[1]:hw[1] + w]
            #     GP_density_map.append(GP_density_map_i)
            # GP_density_map.append(GP_density_map_0)
            # GP_density_map = np.asarray(GP_density_map)
            # GP_density_map = np.reshape(GP_density_map, (1, GP_density_map.shape[0], -1))
        return GP_density_map_0

    def id_unique(self, coords_array):
        # intilize a null list
        coords_array = np.asarray(coords_array)

        unique_list = [[-1, -1, -1, -1]]

        id = coords_array[:, -1]
        n = id.shape[0]
        # traverse for all elements

        for i in range(n):
            id_i = id[i]
            coords_array_i = coords_array[i]

            id_current_unique_list = list(np.asarray(unique_list)[:, -1])
            if id_i not in id_current_unique_list:
                unique_list.append(coords_array_i)

        unique_list = unique_list[1:]
        return unique_list

    def id_diff(self, coords_arrayA, coords_arrayB):  # coords_arrayA is larger
        coords_arrayA = np.asarray(coords_arrayA)
        coords_arrayB = np.asarray(coords_arrayB)

        unique_list = []  # [[-1, -1, -1, -1]]

        if coords_arrayB.size == 0:
            unique_list = coords_arrayA
        else:

            idA = coords_arrayA[:, -1]
            idB = coords_arrayB[:, -1]

            # print(idA)
            # print(idB)

            n = idA.shape[0]
            # traverse for all elements

            for i in range(n):
                id_i = idA[i]
                coords_array_i = coords_arrayA[i]

                # id_current_unique_list = list(np.asarray(unique_list)[:, -1])
                if id_i not in idB:  # id_current_unique_list:
                    unique_list.append(coords_array_i)

            # unique_list = unique_list[1:]
            unique_list = np.asarray(unique_list)
        return unique_list

    def cal_homography_frame(self, v1_pmap_i, v2_pmap_i):
        M = np.asarray([0, 0, -10, 0, 0, -10, 0, 0, 1])
        if v1_pmap_i.shape[0] != 0 and v2_pmap_i.shape[0] != 0:

            if v1_pmap_i.shape[0] >= 4:

                v1_pmap_i[:, 0] = (v1_pmap_i[:, 0] - 0.5) / 0.5
                v1_pmap_i[:, 1] = (v1_pmap_i[:, 1] - 0.5) / 0.5

                v2_pmap_i[:, 0] = (v2_pmap_i[:, 0] - 0.5) / 0.5
                v2_pmap_i[:, 1] = (v2_pmap_i[:, 1] - 0.5) / 0.5

                src_pts = np.float32(v1_pmap_i).reshape(-1, 1, 2)
                dst_pts = np.float32(v2_pmap_i).reshape(-1, 1, 2)

                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, 0, 1.0)  # , cv2.RANSAC)

                except AttributeError:
                    M = np.asarray([0, 0, -10, 0, 0, -10, 0, 0, 1])
        # print(M)
        return M

    def __getitem__(self, index):
        if self.train:
            scene_index = int(index/(self.nb_samplings*100))
        else:
            scene_index = int(index/(self.nb_samplings*100))
        # selection_index = int((index-scene_index*self.nb_samplings*100)/100)

        scene_i = self.scene_name_list[scene_index]
        # scene_i = 'scene_67' # zq

        scene_path = os.path.join(self.file_path, scene_i)
        scene_path_label = os.path.join(self.label_file_path, scene_i)

        # list N frames:
        frame_name_list = os.listdir(scene_path_label)  # CSR-net_multi-view_counting_1output_lowRes_5_5_loadWeights
        if self.train:
            frame_j = random.sample(frame_name_list, k=1)[0]
        else:
            frame_index = (index - scene_index * self.nb_samplings * 100) % 100
            frame_j = frame_name_list[frame_index]

        # select views first:
        frame_0 = '0' # frame_j  # frame_name_list[0]
        label_path0 = os.path.join(self.label_file_path, scene_i, frame_0, 'json_paras/')
        # label_path0 = os.path.join(self.label_file_path, scene_i)
        label_path_list0 = os.listdir(label_path0)
        # here will employ the RL to select view
        label_path_list_sampling = random.sample(label_path_list0, k=self.view_size)
        # for i, sample in enumerate(label_path_list_sampling):
        #     label_path_list_sampling[i] = sample[0:-3] + '.json'
        # label_path_list_sampling = ['11.json', '63.json']
        # for v in range(self.view_size):
        #     view = int(label_path_list_sampling[v].split('.')[0])
        #     while view == 0 or view > 50:
        #         # label_path_list_sampling[v] = str(random.randint(1, 50)) + '.json'
        #         label_path_list_sampling[v] = random.sample(label_path_list0[1:], k=1)[0]
        #         view = int(label_path_list_sampling[v].split('.')[0])

        frame_path = os.path.join(scene_path, frame_j)

        img_path = frame_path + '/jpgs/'
        # # label_path = view_path + '/json_paras/'
        # if self.train:
        label_path = os.path.join(self.label_file_path, scene_i, frame_j, 'json_paras/')
        # label_path = os.path.join(self.label_file_path, scene_i, frame_j, 'jsons/')
        # else:
        #     label_path = os.path.join(self.label_file_path, scene_i, frame_j, 'json_paras/')

        # decide the whole crowd GP density maps of the frame:
        # read all people
        # label_path_list = os.listdir(label_path)
        label_name0 = label_path_list0[0]  #  [0:-3] + '.json'
        label_path_name0 = os.path.join(label_path, label_name0)

        with open(label_path_name0, 'r') as data_file:
            coords_info_frame = json.load(data_file)
        # coords_3d_id_all_frame = read_json_all(coords_info_frame)
        coords_3d_id_all_frame, wld_map_paras_frame, hw_random = self.read_json_frame(coords_info_frame,
                                                                                 self.cropped_size,
                                                                                 self.r, self.a, self.b,
                                                                                 self.patch_num)
        wld_map_paras_frame = np.asarray(wld_map_paras_frame)

        # hw_random = [[194], [215]] # zq
        hw_random = np.asarray(hw_random)
        wld_map_paras = wld_map_paras_frame

        img_views = []
        camera_paras = []
        wld_coords = []

        single_view_dmaps = []
        depth_views = []
        single_view_count = []
        GP_view_dmaps = []

        geted_all_wld_coords = False
        wld_coords_all_of_json = []
        img_coords_nums = []

        homography = []
        M_gt = []
        for idx in range(len(label_path_list_sampling)):
            v1 = label_path_list_sampling[idx]
            map_other = []
            for v2 in label_path_list_sampling:
                if v1 == v2:
                    continue
                label_path1 = os.path.join(label_path, v1)
                label_path2 = os.path.join(label_path, v2)

                with open(label_path1, 'r') as data_file:
                    coords_info1 = json.load(data_file)
                _, v1_pmap, _, _ = self.read_json_view(coords_info1, True)
                with open(label_path2, 'r') as data_file:
                    coords_info2 = json.load(data_file)
                _, v2_pmap, _, _ = self.read_json_view(coords_info2, True)

                v1_pmap = np.asarray(v1_pmap)
                v2_pmap = np.asarray(v2_pmap)
                v1_pmap_12, v2_pmap_12 = [], []
                if v1_pmap.shape[0] != 0 and v2_pmap.shape[0] != 0:
                    id_list1, id_list2 = v1_pmap[:, 2], v2_pmap[:, 2]
                    id_list12 = list(set(id_list1).intersection(id_list2))

                    v1_pmap_12 = [v1_pmap[v1_pmap[:, 2] == id_list12[i], :] for i in range(len(id_list12))]
                    v2_pmap_12 = [v2_pmap[v2_pmap[:, 2] == id_list12[i], :] for i in range(len(id_list12))]


                v1_pmap_12 = np.asarray(v1_pmap_12)
                v2_pmap_12 = np.asarray(v2_pmap_12)

                hfwf = (90, 160)
                dmap = np.zeros(hfwf)
                # tmp = np.zeros(hfwf)
                for i in range(v1_pmap_12.shape[0]):
                    cx = int(v1_pmap_12[i, 0, 0] * 160)
                    cy = int(v1_pmap_12[i, 0, 1] * 90)
                    map_counting(dmap, (cx, cy), sigma=1)

                dmap = np.expand_dims(dmap, axis=0)
                dmap *= 100
                binary_map = np.where(dmap > 0.3, 1, 0)
                map_other.append(binary_map)


                if v1_pmap_12.shape[0] == 0 or v2_pmap_12.shape[0] == 0:
                    matrix_21 = np.asarray([0, 0, -10, 0, 0, -10, 0, 0, 1]).flatten()
                else:
                    matrix_21 = self.cal_homography_frame(v1_pmap_12[:, 0, :2], v2_pmap_12[:, 0, :2]).flatten()
                    matrix_21[8] = 1.0

                homography.append(matrix_21)

                # feature = torch.ones((1, 1, 360, 640))
                # matrix = torch.asarray(matrix_21).reshape((1, 3, 3))
                # output = _transform(feature, matrix)
                # plt.imshow(output[0,0])
                # plt.show()

            M_gt.append(np.expand_dims(np.concatenate(map_other, axis=0), axis=0))

        homography = np.asarray(homography)

        for p in label_path_list_sampling:
            # img_name = p
            # label_name = 'pedInfo' + img_name[5:-6] +'.json'

            label_name = p
            img_name = label_name[0:-5] + '.jpg'
            depthMap_name = label_name[0:-5] + '.h5'

            # read images
            img_path_name = os.path.join(img_path, img_name)
            img = cv2.imread(img_path_name)
            # tmp = cv2.resize(img, (640, 360))

            try:
                img = img[:, :, (2, 1, 0)]  # BGR -> RGB
            except:
                print(img_path_name)

            img = img.astype('float32')

            # normal
            img = img / 255.0
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
            # downsample
            img = cv2.resize(img, (640, 360))

            img_views.append(img)
            # single_view_count.append(tmp)

            # read depthMap:
            # depthMap_path_name = os.path.join(self.depth_file_path, scene_i, depthMap_name)
            # with h5py.File(depthMap_path_name, 'r') as f:
            #     depthMap_i = f['depthMap']
            #     depthMap_i = np.expand_dims(depthMap_i, axis=0)
            # depthMap_i = np.clip(depthMap_i, 0, 1000)
            # # depthMap_i = (depthMap_i - depthMap_i.min()) / (depthMap_i.max() - depthMap_i.min() + 1e-8)
            # depthMap_i = (depthMap_i.max() - depthMap_i) / 100
            # depth_views.append(depthMap_i)

            # read labels
            label_path_name = os.path.join(label_path, label_name)
            with open(label_path_name, 'r') as data_file:
                coords_info = json.load(data_file)
            coords, coords_2d, paras, coords_3d_id_all_of_json = self.read_json_view(coords_info)

            coords_2d_num = len(coords_2d)
            img_coords_nums.append(coords_2d_num)

            coords_2d = np.asarray(coords_2d)
            hfwf = (90, 160)
            single_view_dmaps_i = np.zeros(hfwf)
            # tmp = np.zeros(hfwf)
            for x0, x1 in coords_2d:
                cx = int(x0 * 160)
                cy = int(x1 * 90)
                map_counting(single_view_dmaps_i, (cx, cy), sigma=3)
                # tmp[cy, cx] = 1

            single_view_dmaps_i = np.expand_dims(single_view_dmaps_i, axis=0)
            single_view_dmaps_i *= 100
            single_view_dmaps.append(single_view_dmaps_i)

            # form the camera paras list
            camera_paras.append(paras)

            # get the wld_coords:
            wld_coords = wld_coords + coords


        # get the wld coords
        if len(wld_coords) != 0:
            wld_coords = self.id_unique(wld_coords)
            wld_coords_num = len(wld_coords)
        wld_coords = np.asarray(wld_coords)

        # create the pmaps, instead of density maps
        GP_density_map = self.GP_density_map_creation(wld_coords,
                                                      self.cropped_size,
                                                      wld_map_paras,
                                                      hw_random)

        # GP_density_map *= 100

        img_views = np.asarray(img_views)
        # single_view_count = np.asarray(single_view_count)
        # single_view_count = np.transpose(single_view_count, (0, 3, 1, 2))
        # camera_paras = np.asarray(camera_paras)

        img_views = np.transpose(img_views, (0, 3, 1, 2))
        single_view_dmaps = np.array(single_view_dmaps)
        single_dmap_weight = np.concatenate(M_gt, axis=0)
        # depthMap = np.array(depth_views)

        # print(str(scene_i) + ' ' + str(frame_j) + ' ' + ''.join(label_path_list_sampling))
        if img_views is None:
            print(str(scene_i) + ' ' + str(frame_j) + ' ' + ''.join(label_path_list_sampling))
            assert img_views is not None

        # create distmap
        x_linspace = np.linspace(0, 1., int(single_view_dmaps.shape[3]))
        y_linspace = np.linspace(0, 1., int(single_view_dmaps.shape[2]))
        x_coord, y_coord = np.meshgrid(x_linspace, y_linspace)
        dist = np.sqrt(np.power(x_coord - 0.5, 2) + np.power(y_coord - 1, 2))
        dist = 1 - dist / np.max(dist.flatten())
        dist_map = np.reshape(dist, [single_view_dmaps.shape[2], single_view_dmaps.shape[3]])
        depthMap = np.expand_dims(dist_map, axis=0)
        depthMap = np.expand_dims(depthMap, axis=0)
        depthMap = np.tile(depthMap, [self.num_cam, 1, 1, 1])

        # print(scene_i + " " + frame_j + " " + str(label_path_list_sampling))
        # return img_views, single_view_dmaps, GP_density_map[0,0], homography
        return img_views, single_view_dmaps, GP_density_map, homography, single_dmap_weight, depthMap
        # return img_views, single_view_dmaps, GP_density_map, homography, single_view_count


    def __len__(self):
        # return len(self.map_gt.keys())
        if self.train:
            len_num = self.nb_scenes * 100 * self.nb_samplings
        else:
            len_num = self.nb_scenes * 100 * self.nb_samplings
        # return len(self.map_gt.keys())
        return len_num

def test():
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(MultiviewX(os.path.expanduser('/mnt/d/data/MultiviewX')), train=True, batch_size=2)
    # test projection
    world_grid_maps = []
    xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    import matplotlib.pyplot as plt
    img_views, single_view_dmaps, GP_density_map, homography = dataset.__getitem__(10)
    pass


if __name__ == '__main__':
    test()

