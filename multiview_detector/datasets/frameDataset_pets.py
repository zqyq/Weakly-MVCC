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
# from multiview_detector.utils.random_occlusion import *


class frameDatasetPets(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(), force_download=True,
                 world_reduce=4, map_sigma=5):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root, transform=transform, target_transform=target_transform)


        if train:
            self.batch_size = 2  # batch_size
        else:
            self.batch_size = 1
        self.view_size = 1
        self.patch_num = 1


        self.base = base
        self.root = base.root
        self.img_shape = base.img_shape
        self.train = train
        self.num_cam = base.num_cam
        self.gt_fpath = os.path.join(self.root, 'gt.txt')

        self.hfwf = (288, 384)
        self.hgwg = (768, 576)

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

    def download(self):
        for view in range(self.num_cam):
            imgs = []
            idx = 0
            v_point = {}
            for i, path in enumerate(self.view_path[view]):
                img_path = os.path.join(self.root, 'dmaps', path)
                with h5py.File(img_path, 'r') as f:
                    images_i = f['images']
                    images_i = np.asarray(images_i)
                imgs.append(images_i)

                label_path = os.path.join(self.root, self.img_label_path[i], 'via_region_data_view' + str(view+1) + '.json')
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
            self.imgs[view] = np.concatenate(imgs, axis=0)
            self.imgs_label[view] = v_point

        pass

    def prepare_gt(self):
        gt_list = []
        for path in self.GP_path:
            gt_path = os.path.join(self.root, 'dmaps', path)
            with h5py.File(gt_path, 'r') as f:
                density_map = f['density_maps']
                density_map = np.asarray(density_map)

            gt_list.append(density_map)

        self.ground_plane_gt = np.concatenate(gt_list, axis=0)

        pass

    def corr_map_create(self, frame):
        M_gt = np.zeros((6, 72, 96))
        idxx = 0
        for v1 in range(3):
            for v2 in range(v1, 3):
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

                M_gt[idxx + 2] = np.where(dmap21 > 0.01, 1, 0)
                idxx += 1
        return M_gt


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        imgs = []

        for cam in range(self.num_cam):
            imgs.append(torch.from_numpy(self.imgs[cam][idx]).permute(2, 0, 1))

        imgs = torch.stack(imgs)
        gp_gt = torch.from_numpy(self.ground_plane_gt[idx])

        if self.train:
            M_gt = self.corr_map_create(idx)
            M_gt = np.expand_dims(M_gt, axis=1)
            M_gt = np.reshape(M_gt, (self.num_cam, self.num_cam-1, M_gt.shape[2], -1))
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
        depthMap = np.tile(depthMap, [3, 1, 1, 1])

        return imgs.float(), gp_gt.float(), M_gt, depthMap


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

