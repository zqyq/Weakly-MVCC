import datetime
import os
import json

import cv2
import h5py
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image
from PIL import ImageDraw
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
import numpy as np
from multiview_detector.loss.projasa import *
from .affine_img_demo import data_augmentation
import random


# from multiview_detector.utils.random_occlusion import *


class frameDatasetPets(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(), force_download=True,
                 world_reduce=4, map_sigma=5, seed=14, args=None):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        if train:
            self.batch_size = 2  # batch_size
        else:
            self.batch_size = 1
        self.multi_view_train = args.multi_view_train
        self.view_size = 3 if self.multi_view_train else 1
        self.patch_num = 1
        random.seed(seed)
        self.img_fpaths = base.get_img_fpath(train=train)
        self.density_scale = args.density_scale
        self.augmentation = args.augmentation


        self.base = base
        self.root = base.root
        self.img_shape = base.img_shape
        self.train = train
        self.num_cam = base.num_cam
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.onlyTrainSingleView = args.onlyTrainSingleView

        # self.hfwf = (90, 160)
        self.hgwg = (768, 576)

        self.hfwf = (self.hgwg[1] // 2, self.hgwg[0] // 2)

        # self.len = 2

        if self.train:
            self.view_path = [base.train_view1_frame, base.train_view2_frame, base.train_view3_frame]
            self.GP_path = base.train_GP
            self.img_label_path = ['PETS2009_labels/S1L3/14_17/', 'PETS2009_labels/S1L3/14_33/',
                                   'PETS2009_labels/S2L2/14_55/', 'PETS2009_labels/S2L3/14_41/']
            self.abandon_list = ['frame_0433.jpg', 'frame_0434.jpg', 'frame_0435.jpg']
            self.len = 1105

        else:
            self.view_path = [base.test_view1_frame, base.test_view2_frame, base.test_view3_frame]
            self.GP_path = base.test_GP
            self.img_label_path = ['PETS2009_labels/S1L1/13_57/', 'PETS2009_labels/S1L1/13_59/',
                                   'PETS2009_labels/S1L2/14_06/', 'PETS2009_labels/S1L2/14_31/']
            self.abandon_list = []
            self.len = 794

        self.imgs = {view: {} for view in range(3)}
        self.imgs_label = {view: {} for view in range(4)}
        self.ground_plane_gt = []

        self.download()
        self.prepare_gt()

    def affine_img(self, img):
        '''

        :param img (H, W, C):
        :return: affined img
        '''


    def download(self):
        for view in range(self.num_cam):
            imgs = []
            idx = 0
            W, H = self.img_shape
            v_point = {}
            for i, path in enumerate(self.view_path[view]):
                if idx == 2:
                    break
                img_path = os.path.join(self.root, 'dmaps', path)
                with h5py.File(img_path, 'r') as f:
                    images_i = f['images']
                    images_i = np.asarray(images_i)
                imgs.append(images_i)

                label_path = os.path.join(self.root, self.img_label_path[i],
                                          'via_region_data_view' + str(view + 1) + '.json')
                with open(label_path, 'r') as data_file:
                    v1_pmap = json.load(data_file)

                for key in v1_pmap.keys():
                    if self.img_label_path[i] == 'PETS2009_labels/S2L2/14_55/':
                        if key in self.abandon_list:
                            continue
                    v_pi = []
                    regions = v1_pmap[key]['regions']
                    for point_id in regions:
                        point_id_num = int(float(point_id))
                        cx = int(regions[point_id]['shape_attributes']['cx'])
                        cy = int(regions[point_id]['shape_attributes']['cy'])
                        if cx < 0 or cx > 768 or cy < 0 or cy > 576 or (cx < 50 and cy < 50):
                            continue
                        corrds_2d = [int(cy / 8), int(cx / 8)] + [point_id_num]
                        v_pi.append(corrds_2d)

                    v_point[idx] = v_pi
                    idx += 1
                pass

            # count = 0

            # for frame_idx, img_i_path in self.img_fpaths[view + 1].items():
            #     if count == 2:
            #         break
            #     count += 1
            #
            #     # plt.imshow(img)
            #     # plt.show()
            #     self.imgs[view][frame_idx] = img
            self.imgs[view] = np.concatenate(imgs, axis=0)
            self.imgs_label[view] = v_point
            # continue # debug, one frame

        pass

    def read_img_by_path(self, img_i_path, scale=2):
        img = cv2.imread(img_i_path)
        # tmp = cv2.resize(img, (640, 360))
        try:
            img = img[:, :, (2, 1, 0)]  # BGR -> RGB
        except:
            print(img_i_path)
        img = img.astype(np.float32)
        # normal
        img = img / 255.0
        # img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        # img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        # img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        # downsample
        H, W, C = img.shape
        img = cv2.resize(img, (W // scale, H // scale))
        return img

    def prepare_gt(self):
        gt_list = []
        for path in self.GP_path:
            gt_path = os.path.join(self.root, 'dmaps', path)
            with h5py.File(gt_path, 'r') as f:
                density_map = f['density_maps']
                density_map = np.asarray(density_map)

            gt_list.append(density_map)
            # continue # debug, one frame
        self.ground_plane_gt = np.concatenate(gt_list, axis=0)

        pass

    def corr_map_create(self, frame):
        M_gt = np.zeros((6, 72, 96))
        idxx = 0
        for v1 in range(3):
            for v2 in range(v1, 3):
                '''
                
                0, 1
                0, 2
                1, 2
                
                '''
                if v2 == v1:
                    continue
                v1_point = np.asarray(self.imgs_label[v1][frame])
                v2_point = np.asarray(self.imgs_label[v2][frame])

                id_list1, id_list2 = v1_point[:, 2], v2_point[:, 2]
                id_list12 = list(set(id_list1).intersection(id_list2))

                v1_point_12 = [v1_point[v1_point[:, 2] == id_list12[i], :] for i in range(len(id_list12))]
                v2_point_12 = [v2_point[v2_point[:, 2] == id_list12[i], :] for i in range(len(id_list12))]

                v1_point_12 = np.asarray(v1_point_12)
                v2_point_12 = np.asarray(v2_point_12)

                dmap12 = np.zeros((72, 96))
                for i in range(v1_point_12.shape[0]):
                    cx = int(v1_point_12[i, 0, 0])
                    cy = int(v1_point_12[i, 0, 1])
                    map_counting(dmap12, (cy, cx), sigma=2)

                M_gt[idxx] = np.where(dmap12 > 0.01, 1, 0)

                dmap21 = np.zeros((72, 96))
                for i in range(v2_point_12.shape[0]):
                    cx = int(v2_point_12[i, 0, 0])
                    cy = int(v2_point_12[i, 0, 1])
                    map_counting(dmap21, (cy, cx), sigma=2)
                M_gt[idxx + 3] = np.where(dmap21 > 0.01, 1, 0)
                idxx += 1
                '''
                    0: 0, 1
                    1: 0, 2
                    2: 1, 2
                    3: 1, 0
                    4: 2, 0
                    5: 2, 1
                '''
        M_gt[2], M_gt[3] = M_gt[3], M_gt[2]
        '''
                    0: 0, 1
                    1: 0, 2
                    2: 1, 0
                    3: 1, 2
                    4: 2, 0
                    5: 2, 1        

        '''
        return M_gt

    def __len__(self):
        return self.len
        # return 20

    def __getitem__(self, idx):
        img_views = []


        hfwf = self.hfwf
        H, W = hfwf
        # H, W = H // 2, W // 2
        scale = 4  # scale of img density map  # (144, 192)
        img_dmap_shape = [H // scale, W // scale]
        # patch_W1 = 192
        # patch_W2 = 160
        # patch_W2_Padding = 32
        patch_W1 = 256
        patch_W2 = 224
        patch_W2_Padding = 32
        patch_W3 = 192
        patch_W3_Padding = 64

        seed = -1

        single_view_dmaps = []

        all_random = []

        selectedView = -1
        use_rank_mask = self.train and not self.multi_view_train

        imgs_single_C = []

        if self.onlyTrainSingleView and self.train:
            selectedView = random.randint(0, self.num_cam - 1)

        for cam in range(self.num_cam):
            if selectedView != -1 and cam != selectedView:
                continue
            # if self.train:

            imgh_range = range(0, H - patch_W1 + 1)  # 192 大小的块, 或者 192 / 2
            imgw_range = range(0, W - patch_W1 + 1)
            imgh_random = random.sample(imgh_range, k=2)
            imgw_random = random.sample(imgw_range, k=2)
            all_random.append([imgh_random, imgw_random])

            # plt.imshow(self.imgs[cam][idx])
            # plt.show()
            hfwf = img_dmap_shape
            single_view_dmaps_i = np.zeros(hfwf)
            mask1_dmaps_i = np.zeros(hfwf)
            mask2_dmaps_i = np.zeros(hfwf)
            mask3_dmaps_i = np.zeros(hfwf)
            mask4_dmaps_i = np.zeros(hfwf)
            mask5_dmaps_i = np.zeros(hfwf)
            mask6_dmaps_i = np.zeros(hfwf)

            # plt.imshow(single_view_dmaps_i)
            # plt.show()

            # tmp = np.zeros(hfwf)
            use_augmentation = False
            if self.train and self.augmentation:
                use_augmentation = random.randint(0, 1) == 0

            for cy, cx, id in self.imgs_label[cam][idx]:
                # cx = int(x0 * 160)
                # cy = int(x1 * 90)
                cy = cy * 4  # img shape: (768 // 2, 576 // 2)： （288, 384）
                cx = cx * 4  # img_density_map: (768 // 4, 576 //4): (144, 192)

                map_counting(single_view_dmaps_i, (cx // scale, cy // scale), sigma=1)

                if use_rank_mask:

                    ci = int(cx)
                    cj = int(cy)
                    if ci >= 0 and ci < W and cj >= 0 and cj < H:
                        if (ci >= imgw_random[0] and ci <= imgw_random[0] + patch_W1 and cj >= imgh_random[0] and cj <=
                                imgh_random[0] + patch_W1):  # ranking
                            # mask1_dmaps_i[int(cj // scale), int(ci // scale)] = 1
                            map_counting(mask1_dmaps_i, (ci // scale, cj // scale), sigma=3)
                        if (ci >= imgw_random[1] and ci <= imgw_random[1] + patch_W1 and cj >= imgh_random[1] and cj <=
                                imgh_random[1] + patch_W1):
                            map_counting(mask2_dmaps_i, (ci // scale, cj // scale), sigma=3)
                            # mask2_dmaps_i[int(cj // scale), int(ci // scale)] = 1

                        if (ci >= imgw_random[0] + patch_W2_Padding and ci <= imgw_random[0] + patch_W2 and cj >= imgh_random[
                            0] + patch_W2_Padding and cj <= imgh_random[0] + patch_W2):
                            # mask3_dmaps_i[int(cj // scale), int(ci // scale)] = 1
                            map_counting(mask3_dmaps_i, (ci // scale, cj // scale), sigma=3)
                        if (ci >= imgw_random[1] + patch_W2_Padding and ci <= imgw_random[1] + patch_W2 and cj >= imgh_random[
                            1] + patch_W2_Padding and cj <= imgh_random[1] + patch_W2):
                            # mask4_dmaps_i[int(cj // scale), int(ci // scale)] = 1
                            map_counting(mask4_dmaps_i, (ci // scale, cj // scale), sigma=3)

                        if (ci >= imgw_random[0] + patch_W3_Padding and ci <= imgw_random[0] + patch_W3 and cj >= imgh_random[
                            0] + patch_W3_Padding and cj <= imgh_random[0] + patch_W3):
                            # mask5_dmaps_i[int(cj // scale), int(ci // scale)] = 1
                            map_counting(mask5_dmaps_i, (ci // scale, cj // scale), sigma=3)
                        if (ci >= imgw_random[1] + patch_W3_Padding and ci <= imgw_random[1] + patch_W3 and cj >= imgh_random[
                            1] + patch_W3_Padding and cj <= imgh_random[1] + patch_W3):
                            # mask6_dmaps_i[int(cj // scale), int(ci // scale)] = 1
                            map_counting(mask6_dmaps_i, (ci // scale, cj // scale), sigma=3)

                # tmp[cy, cx] = 1
            if self.augmentation:
                if use_augmentation:
                    single_view_dmaps_i, seed = data_augmentation(single_view_dmaps_i, is_density_map=True)
                    # aug_seed.append(seed)
                    if use_rank_mask:
                        mask1_dmaps_i, _ = data_augmentation(mask1_dmaps_i, seed=seed, is_density_map=True)
                        mask2_dmaps_i, _ = data_augmentation(mask2_dmaps_i, seed=seed, is_density_map=True)
                        mask3_dmaps_i, _ = data_augmentation(mask3_dmaps_i, seed=seed, is_density_map=True)
                        mask4_dmaps_i, _ = data_augmentation(mask4_dmaps_i, seed=seed, is_density_map=True)
                        mask5_dmaps_i, _ = data_augmentation(mask5_dmaps_i, seed=seed, is_density_map=True)
                        mask6_dmaps_i, _ = data_augmentation(mask6_dmaps_i, seed=seed, is_density_map=True)

            single_view_dmaps_i = np.expand_dims(single_view_dmaps_i, axis=0)
            single_view_dmaps.append(single_view_dmaps_i)

            if use_rank_mask:
                mask1_dmaps_i = np.expand_dims(mask1_dmaps_i, axis=0)
                mask2_dmaps_i = np.expand_dims(mask2_dmaps_i, axis=0)
                mask3_dmaps_i = np.expand_dims(mask3_dmaps_i, axis=0)
                mask4_dmaps_i = np.expand_dims(mask4_dmaps_i, axis=0)
                mask5_dmaps_i = np.expand_dims(mask5_dmaps_i, axis=0)
                mask6_dmaps_i = np.expand_dims(mask6_dmaps_i, axis=0)
                # single_view_dmaps_i *= 100
                single_view_dmaps.append(mask1_dmaps_i)
                single_view_dmaps.append(mask2_dmaps_i)
                single_view_dmaps.append(mask3_dmaps_i)
                single_view_dmaps.append(mask4_dmaps_i)
                single_view_dmaps.append(mask5_dmaps_i)
                single_view_dmaps.append(mask6_dmaps_i)

            img_path = self.img_fpaths[cam + 1][idx]
            img = self.read_img_by_path(img_path)

            # plt.imshow(single_view_dmaps_i[0])
            # plt.show()
            #
            # plt.imshow(img)
            # plt.show()
            # print(img.shape)
            if self.augmentation:
                if use_augmentation:
                    img, _ = data_augmentation(img, seed=seed)
            img = img.transpose(2, 0, 1)[None]

            # plt.imshow(img)

            img_views.append(img)
            imgs_single_C.append(torch.from_numpy(self.imgs[cam][idx]).permute(2, 0, 1))

        # imgs = torch.stack(imgs)  # (V, C, H, W)
        # img_views = np.asarray(img_views)
        # img_views = np.transpose(img_views, (0, 3, 1, 2))

        # single_view_dmaps = np.stack(single_view_dmaps)

        all_imgs = []
        if not use_rank_mask:
            all_imgs = img_views
        else:
            for view_i, random_i in enumerate(all_random):
                mask1 = np.zeros((1, 1, H, W))
                mask2 = np.zeros((1, 1, H, W))
                mask3 = np.zeros((1, 1, H, W))
                mask4 = np.zeros((1, 1, H, W))
                mask5 = np.zeros((1, 1, H, W))
                mask6 = np.zeros((1, 1, H, W))

                imgh_random, imgw_random = random_i
                img_i = img_views[view_i]
                mask1[:, :, imgh_random[0]:imgh_random[0] + patch_W1, imgw_random[0]:imgw_random[
                                                                                      0] + patch_W1] = 1  # 192 contains 128, i.e. mask1 contains mask3, mask2 contains mask4
                mask1_img = img_i.copy()
                # mask1_img[:, :, imgh_random:imgh_random+192, imgw_random:imgw_random+192] = mask1
                mask1_img = np.multiply(mask1_img, mask1)
                # mask1_img_list.append(mask1_img)

                # plt.imshow(mask1_img.squeeze().transpose(1, 2, 0))
                # plt.show()
                #
                # plt.imshow(mask1.squeeze())
                # plt.show()

                mask2[:, :, imgh_random[1]:imgh_random[1] + patch_W1, imgw_random[1]:imgw_random[1] + patch_W1] = 1
                mask2_img = img_i.copy()
                # mask1_img[:, :, imgh_random:imgh_random+192, imgw_random:imgw_random+192] = mask1
                mask2_img = np.multiply(mask2_img, mask2)
                # mask2_img_list.append(mask2_img)

                mask3_img = img_i.copy()
                # mask2_img[:, :, imgh_random+32:imgh_random+160, imgw_random+32:imgw_random+160] = mask2
                mask3[:, :, imgh_random[0] + patch_W2_Padding:imgh_random[0] + patch_W2,
                imgw_random[0] + patch_W2_Padding:imgw_random[0] + patch_W2] = 1
                mask3_img = np.multiply(mask3_img, mask3)
                # mask3_img_list.append(mask3_img)

                mask4_img = img_i.copy()
                # mask2_img[:, :, imgh_random+32:imgh_random+160, imgw_random+32:imgw_random+160] = mask2
                mask4[:, :, imgh_random[1] + patch_W2_Padding:imgh_random[1] + patch_W2,
                imgw_random[1] + patch_W2_Padding:imgw_random[1] + patch_W2] = 1
                mask4_img = np.multiply(mask4_img, mask4)
                # mask4_img_list.append(mask4_img)

                mask5_img = img_i.copy()
                # mask2_img[:, :, imgh_random+32:imgh_random+160, imgw_random+32:imgw_random+160] = mask2
                mask5[:, :, imgh_random[0] + patch_W3_Padding:imgh_random[0] + patch_W3,
                imgw_random[0] + patch_W3_Padding:imgw_random[0] + patch_W3] = 1
                mask5_img = np.multiply(mask5_img, mask5)

                mask6_img = img_i.copy()
                # mask2_img[:, :, imgh_random+32:imgh_random+160, imgw_random+32:imgw_random+160] = mask2
                mask6[:, :, imgh_random[1] + patch_W3_Padding:imgh_random[1] + patch_W3,
                imgw_random[1] + patch_W3_Padding:imgw_random[1] + patch_W3] = 1
                mask6_img = np.multiply(mask6_img, mask6)

                all_imgs.append(img_i)
                all_imgs.append(mask1_img)
                all_imgs.append(mask2_img)
                all_imgs.append(mask3_img)
                all_imgs.append(mask4_img)
                all_imgs.append(mask5_img)
                all_imgs.append(mask6_img)

        # (Vi, Vi_mask1, Vi_mask2, ...Vi_mask4, ...)

        img_views = np.concatenate(all_imgs, axis=0)
        imgs_single_C = np.concatenate(imgs_single_C, axis=0)
        V, H, W = imgs_single_C.shape
        imgs_single_C = imgs_single_C.reshape(V, 1, H, W)
        try:
            single_view_dmaps = np.array(single_view_dmaps)
        except:
            pass

        # rows, col = 3, 7
        # fig, axes = plt.subplots(nrows=rows, ncols=col, figsize=(16, 8))  # figsize 控制画布大小
        # # 遍历所有子图并绘图
        # for i in range(rows):
        #     for j in range(col):
        #         ax = axes[i, j]  # 获取当前子图
        #         # ax.imshow(img_views.reshape((3, -1, 3, H, W))[i, j].transpose(1, 2, 0))
        #         ax.imshow(single_view_dmaps.reshape((3, -1, 1, 96, 128))[i, j].transpose(1, 2, 0))
        # # 调整子图间距
        # plt.tight_layout()
        # # plt.savefig(os.path.join(self.logdir, f'imgs_{epoch}_{batch_idx}.png'))
        # plt.show()

        single_view_dmaps *= self.density_scale

        gp_gt = torch.from_numpy(self.ground_plane_gt[idx])  # (710, 610, 1)
        if self.train:
            M_gt = self.corr_map_create(idx)
            M_gt = np.expand_dims(M_gt, axis=1)
            M_gt = np.reshape(M_gt, (self.num_cam, self.num_cam - 1, M_gt.shape[2], -1))  # (B, C, H / 4, W / 4), (3, 2, 72, 96
        else:
            M_gt = 0

        x_linspace = np.linspace(0, 1., int(self.hfwf[1] / 4))
        y_linspace = np.linspace(0, 1., int(self.hfwf[0] / 4))
        x_coord, y_coord = np.meshgrid(x_linspace, y_linspace)
        dist = np.sqrt(np.power(x_coord - 0.5, 2) + np.power(y_coord - 1, 2))
        dist = 1 - dist / np.max(dist.flatten())
        dist_map = np.reshape(dist, [int(self.hfwf[0] / 4), int(self.hfwf[1] / 4)])
        depthMap = np.expand_dims(dist_map, axis=0)
        depthMap = np.expand_dims(depthMap, axis=0)
        depthMap = np.tile(depthMap, [3, 1, 1, 1])  # [3, 1, 72, 96], [B, C, H / 4, W / 4]

        # plt.imshow(M_gt[0][0])
        # plt.show()

        return img_views, gp_gt.float(), M_gt, depthMap, single_view_dmaps, imgs_single_C  # (img_gray, ground_plane_densityMap, singleViewMask, depthMap)


def test():
    from multiview_detector.datasets.Pets import Pets
    import torch.nn.functional as F

    # transform = T.Compose([T.Resize([760, 1352]),  # H,W
    #                        T.ToTensor(),
    #                        T.Normalize((0.4424, 0.4292, 0.4089), (0.2500, 0.2599, 0.2549))])
    # dataset = frameDataset(Citystreet(os.path.expanduser('~/Data/CityStreet')), train=True, map_sigma=5, img_sigma=3)
    # dataloader = Dataloader(dataset,1,False,num_workers=0)
    dataset_train = frameDataset(Pets(os.path.expanduser('/mnt/d/data/PETS2009/')), train=True, map_sigma=5,
                                 world_reduce=2)
    dataset_test = frameDataset(Pets(os.path.expanduser('/mnt/d/data/PETS2009/')), train=False, map_sigma=5,
                                world_reduce=2)
    # for i in range(300):
    #     imgs, map_gt, imgs_gt, frame = dataset_train.__getitem__(i)
    #     for view in range(3):
    #         img_view_gt = imgs_gt[view]
    #         if img_view_gt.sum() == 0:
    #             print(f'view:{view}, frame:{frame}')
    imgs, map_gt, imgs_gt, frame = dataset_train.__getitem__(10)
    # loginfo = open('test_info.txt', 'a')
    # sys.stdout = loginfo
    print(f'imgs shape= {imgs.shape}')
    for view in range(3):
        plt.imshow(imgs[view].permute([1, 2, 0]))
        plt.show()
    print('map_gt shape', map_gt.shape)
    print('img_gt shape', imgs_gt[0].shape)
    print('sum of imgs-gt[0]', imgs_gt[0].sum().item())
    # print('img max', imgs_gt[0].max())
    # print('kernel shape==', dataset.map_kernel.shape)
    #
    map_gt = F.conv2d(map_gt.detach().unsqueeze(0), dataset_train.map_kernel.float(),
                      padding=int((dataset_train.map_kernel.shape[-1] - 1) / 2))
    # img_gt0 = F.adaptive_max_pool2d(imgs_gt[0][None], (380, 676))
    # img_gt0 = F.conv2d(img_gt0, dataset_test.img_kernel.float(),
    #                    padding=int((dataset_test.img_kernel.shape[-1] - 1) / 2))
    plt.imshow(map_gt.squeeze())
    plt.show()
    # plt.imshow(imgs_gt[0].squeeze())
    # plt.show()
    print('map max', map_gt.max())
    print('img max', imgs_gt[0].max())
    print(datetime.datetime.now(), '\n')
    # loginfo.close()


if __name__ == '__main__':
    test()
