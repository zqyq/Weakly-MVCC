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


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(), force_download=True,
                 world_reduce=4, map_sigma=5):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = map_sigma, 4 * map_sigma
        self.img_shape = base.img_shape
        self.train = train
        self.transform = transform
        self.base = base
        self.root, self.num_frame = base.root, base.num_frame
        self.num_cam = base.num_cam
        self.hfwf = (380, 676)
        self.hgwg = (768, 640)
        self.world_reduce = world_reduce
        self.Rworld_shape = tuple(map(lambda x: x // world_reduce, self.hgwg))
        frame_rangelist = [range(636, 1236, 2), range(1236, 1636, 2)]
        if self.train:
            frame_range = frame_rangelist[0]
        else:
            frame_range = frame_rangelist[1]
        self.img_fpaths = self.base.get_img_fpath()
        # self.map_gt = {}
        self.ground_plane_gt = {}
        # self.view_gts = {view: {} for view in range(3)}
        self.imgs_head_gt = {view: {} for view in range(1, 4)}
        self.imgs_read = {view: {} for view in range(1, 4)}
        self.imgs_mask = {view: {} for view in range(1, 4)}
        self.imgs_label = {view: {} for view in range(1, 4)}

        self.read_mask_npz()
        self.download(frame_range)  # 获得map_gt,和 imgs_gt

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel /= map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

    def read_mask_npz(self):
        aimdir = self.get_dmaps_path()
        mask1 = np.load(aimdir['v1_img_mask'])
        self.imgs_mask[1] = mask1.f.arr_0
        mask2 = np.load(aimdir['v2_img_mask'])
        self.imgs_mask[2] = mask2.f.arr_0
        mask3 = np.load(aimdir['v3_img_mask'])
        self.imgs_mask[3] = mask3.f.arr_0


    def get_dmaps_path(self):
        aimdir = {
            'gp_train': os.path.join(self.root, 'GT_density_maps/'
                                                'ground_plane/train/Street_groundplane_train_dmaps_10.h5'),
            'gp_test': os.path.join(self.root, 'GT_density_maps/'
                                               'ground_plane/test/Street_groundplane_test_dmaps_10.h5'),
            'v1_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view1_dmap_10.h5'),
            'v2_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view2_dmap_10.h5'),
            'v3_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view3_dmap_10.h5'),

            'v1_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view1_dmap_10.h5'),
            'v2_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view2_dmap_10.h5'),
            'v3_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view3_dmap_10.h5'),

            'v1_img_label': os.path.join(self.root, 'labels/via_region_data_view1.json'),
            'v2_img_label': os.path.join(self.root, 'labels/via_region_data_view2.json'),
            'v3_img_label': os.path.join(self.root, 'labels/via_region_data_view3.json'),

            'v1_img_mask': os.path.join(self.root, 'ROI_maps/ROIs/camera_view/mask1_ic.npz'),
            'v2_img_mask': os.path.join(self.root, 'ROI_maps/ROIs/camera_view/mask2_ic.npz'),
            'v3_img_mask': os.path.join(self.root, 'ROI_maps/ROIs/camera_view/mask3_ic.npz') }

        return aimdir

    def load_json(self, jsondir, mask, frame_range_list):
        with open(jsondir) as data_file:
            v_pmap_json = json.load(data_file)

        v_point = {}
        h0, w0 = self.hfwf

        for key in v_pmap_json.keys():
            img_id = int(key[6:10])
            regions = v_pmap_json[key]['regions']

            v_pi = []

            for point_id in regions:
                point_id_num = int(float(point_id))
                # whole_id = regions[point_id]['region_attributes']['whole_ID']
                try:
                    cx = int(regions[point_id]['shape_attributes']['cx'])
                    cy = int(regions[point_id]['shape_attributes']['cy'])
                except TypeError:
                    continue
                if cx < 0 or cx >= w0 * 4 or cy < 0 or cy >= h0 * 4 or (mask[cy, cx] == 0):
                    continue
                else:
                    # v1_pmapi = [img_id, point_id_num, int(cx/4), int(cy/4)]
                    # v1_pmap.append(v1_pmapi)
                    corrds_2d = [int(cy / 16), int(cx / 16)] + [point_id_num]
                    v_pi.append(corrds_2d)

            v_point[img_id] = v_pi

        # v_point = np.asarray(v_point)
        return v_point


    def load_h5(self, h5dir, frame_range_list):
        temp_gt = {}
        temp_img = {}
        with h5py.File(h5dir, 'r') as fp:
            dmap_i = fp['density_maps']
            dmap_i = np.squeeze(dmap_i).astype(np.float32)
            # print('dmap_i shape', dmap_i.shape)

            image_i = np.transpose(fp['images'], (0, 3, 1, 2)).astype(np.float32)

            for i in range(0, dmap_i.shape[0]):
                temp_gt[frame_range_list[i]] = dmap_i[i][:][:]
                temp_img[frame_range_list[i]] = image_i[i]

        return temp_gt, temp_img

    def prepare_gt(self):
        og_gt = []
        with h5py.File(os.path.expanduser('/mnt/d/data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
            for i in range(f['v_pmap_GP'].shape[0]):
                singlePerson_Underframe = f['v_pmap_GP'][i]
                frame = int(singlePerson_Underframe[0])
                # personID = int(singlePerson_Underframe[1])
                # 原论文grid_x 为H这条边，singleFrame_underframe[3]指的是cy，最大值不超过768
                # 原论文grid_y 为W这条边，singleFrame_underframe[2]指的是cx，最大值不超过640
                grid_y = int(singlePerson_Underframe[2] * 4)  # 乘以4之后，每个pixel代表2.5cm,
                grid_x = int(singlePerson_Underframe[3] * 4)  # [3072, 2560]
                # height = int(singlePerson_Underframe[4])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        aimdir = self.get_dmaps_path()
        bbox_by_avesize = {}
        # map_gt [768,640]
        with h5py.File(os.path.expanduser('/mnt/d/data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
            grounddata = f['v_pmap_GP']
            for frame in frame_range:
                occupancy_info = (grounddata[grounddata[:, 0] == frame, 2:4])
                occupancy_map = torch.zeros(size=self.Rworld_shape, requires_grad=False)
                for idx in range(occupancy_info.shape[0]):
                    cx, cy = occupancy_info[idx]
                    cx = int(cx / self.world_reduce)
                    cy = int(cy / self.world_reduce)
                    occupancy_map[cy, cx] = 1
                occupancy_map = occupancy_map.unsqueeze(0)
                self.ground_plane_gt[frame] = occupancy_map

        # imgs_gt [380,676]
        if self.train:
            # temp_gp_train = self.load_h5(aimdir['gp_train'], frame_range)
            temp_view1_train = self.load_h5(aimdir['v1_train'], frame_range)
            temp_view2_train = self.load_h5(aimdir['v2_train'], frame_range)
            temp_view3_train = self.load_h5(aimdir['v3_train'], frame_range)

            view1_img_label = self.load_json(aimdir['v1_img_label'], self.imgs_mask[1], frame_range)
            view2_img_label = self.load_json(aimdir['v2_img_label'], self.imgs_mask[2], frame_range)
            view3_img_label = self.load_json(aimdir['v3_img_label'], self.imgs_mask[3], frame_range)

            for i in frame_range:
                # tmp_mapgt = temp_gp_train[i]
                # tmp_mapgt = (tmp_mapgt - np.min(tmp_mapgt)) / np.max(tmp_mapgt)
                # self.map_gt[i] = tmp_mapgt
                self.imgs_head_gt[1][i], self.imgs_read[1][i] = temp_view1_train[0][i], temp_view1_train[1][i]
                self.imgs_head_gt[2][i], self.imgs_read[2][i] = temp_view2_train[0][i], temp_view2_train[1][i]
                self.imgs_head_gt[3][i], self.imgs_read[3][i] = temp_view3_train[0][i], temp_view3_train[1][i]

                self.imgs_label[1][i] = view1_img_label[i]
                self.imgs_label[2][i] = view2_img_label[i]
                self.imgs_label[3][i] = view3_img_label[i]
        else:
            # temp_gp_test = self.load_h5(aimdir['gp_test'], frame_range)
            temp_view1_test = self.load_h5(aimdir['v1_test'], frame_range)
            temp_view2_test = self.load_h5(aimdir['v2_test'], frame_range)
            temp_view3_test = self.load_h5(aimdir['v3_test'], frame_range)
            for i in frame_range:
                # tmp_mapgt = temp_gp_test[i]
                # tmp_mapgt = (tmp_mapgt - np.min(tmp_mapgt)) / np.max(tmp_mapgt)
                # self.map_gt[i] = tmp_mapgt
                self.imgs_head_gt[1][i], self.imgs_read[1][i] = temp_view1_test[0][i], temp_view1_test[1][i]
                self.imgs_head_gt[2][i], self.imgs_read[2][i] = temp_view2_test[0][i], temp_view2_test[1][i]
                self.imgs_head_gt[3][i], self.imgs_read[3][i] = temp_view3_test[0][i], temp_view3_test[1][i],

    def corr_map_create(self, frame):
        M_gt = np.zeros((6, 47, 84))
        idxx = 0
        for v1 in range(1, 4):
            for v2 in range(v1, 4):
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

                dmap12 = np.zeros((47, 84))
                for i in range(v1_point_12.shape[0]):
                    cx = int(v1_point_12[i, 0, 0] / 2)
                    cy = int(v1_point_12[i, 0, 1] / 2)
                    map_counting(dmap12, (cy, cx), sigma=1)

                M_gt[idxx] = np.where(dmap12*10 > 0.01, 1, 0)

                dmap21 = np.zeros((47, 84))
                for i in range(v2_point_12.shape[0]):
                    cx = int(v2_point_12[i, 0, 0] / 2)
                    cy = int(v2_point_12[i, 0, 1] / 2)
                    map_counting(dmap21, (cy, cx), sigma=1)

                M_gt[idxx + 2] = np.where(dmap21*10 > 0.01, 1, 0)
                idxx += 1
        return M_gt

    def __getitem__(self, index):
        frame = list(self.ground_plane_gt.keys())[index]
        imgs = []

        # generate random occlusions
        # if self.transform is not None and self.train is True:
        #     random_list = generate_occlusion_list(self.base, 0, 768 * 640 - 1, 25)  # 768 * 640 - 1=491519

        for cam in range(self.num_cam):
            # fpath = self.img_fpaths[cam + 1][frame]
            # # img = Image.open(fpath).convert('RGB')
            # img = cv2.imread(fpath)
            # img = cv2.resize(img, [self.hfwf[1], self.hfwf[0]])
            # # imga = ImageDraw.ImageDraw(img)
            #
            # # if self.transform is not None and self.train is True:
            # #     for pos in random_list:
            # #         bbox = get_rect(cam + 1, pos)
            # #         if bbox is not None:
            # #             imga.rectangle((tuple(bbox[:2]), tuple(bbox[2:])), fill='gray', outline=None, width=1)
            # if self.transform is not None:
            #     img = self.transform(img)

            img = self.imgs_read[cam + 1][frame]
            imgs.append(torch.from_numpy(img))
        imgs = torch.stack(imgs)
        gp_gt = self.ground_plane_gt[frame]

        imgs_gt = []
        for view in range(1, 4):
            img_gt = self.imgs_head_gt[view][frame]
            imgs_gt.append(torch.from_numpy(img_gt)[None])

        if self.train:
            M_gt = self.corr_map_create(frame)
            M_gt = np.expand_dims(M_gt, axis=1)
            M_gt = np.reshape(M_gt, (self.num_cam, self.num_cam-1, M_gt.shape[2], -1))
        else:
            M_gt = 0

        x_linspace = np.linspace(0, 1., int(self.hfwf[1] / 4))
        y_linspace = np.linspace(0, 1., int(self.hfwf[0] / 4))
        x_coord, y_coord = np.meshgrid(x_linspace, y_linspace)
        dist = np.sqrt(np.power(x_coord - 0.5, 2) + np.power(y_coord - 1, 2))
        dist = 1 - dist / np.max(dist.flatten())
        dist_map = np.reshape(dist, [95, 169])
        depthMap = np.expand_dims(dist_map, axis=0)
        depthMap = np.expand_dims(depthMap, axis=0)
        depthMap = np.tile(depthMap, [3, 1, 1, 1])
        
        # return  imgs, imgs_gt, gp_gt.float()
        return imgs, imgs_gt, gp_gt.float(), frame, M_gt, depthMap

    def __len__(self):
        return len(self.ground_plane_gt.keys())


def test():
    from multiview_detector.datasets.Citystreet import Citystreet
    import torch.nn.functional as F

    # transform = T.Compose([T.Resize([760, 1352]),  # H,W
    #                        T.ToTensor(),
    #                        T.Normalize((0.4424, 0.4292, 0.4089), (0.2500, 0.2599, 0.2549))])
    # dataset = frameDataset(Citystreet(os.path.expanduser('~/Data/CityStreet')), train=True, map_sigma=5, img_sigma=3)
    # dataloader = Dataloader(dataset,1,False,num_workers=0)
    dataset_train = frameDataset(Citystreet(os.path.expanduser('/mnt/d/data/CityStreet')), train=True, map_sigma=5,
                                 world_reduce=2)
    dataset_test = frameDataset(Citystreet(os.path.expanduser('/mnt/d/data/CityStreet')), train=False, map_sigma=5,
                                world_reduce=2)
    # for i in range(300):
    #     imgs, map_gt, imgs_gt, frame = dataset_train.__getitem__(i)
    #     for view in range(3):
    #         img_view_gt = imgs_gt[view]
    #         if img_view_gt.sum() == 0:
    #             print(f'view:{view}, frame:{frame}')
    imgs, map_gt, imgs_gt, frame = dataset_train.__getitem__(0)
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
