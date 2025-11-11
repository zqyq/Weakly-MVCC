import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from multiview_detector.models.correlation_layer import Correlation_Layer
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector_homo(nn.Module):
    def __init__(self, dataset, arch='vgg11'):  # vgg11 resnet18
        super().__init__()

        # end--------------------
        self.batch_size = dataset.batch_size
        self.view_size = dataset.view_size
        self.patch_num = dataset.patch_num  # it is used to control the number of sample block in single image
        self.output_size = [160, 180]
        self.reducedgrid_shape = [160, 180]

        self.device = ["cuda:0", "cuda:1"]
        self.num_cam = dataset.num_cam

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        out_channel = 256

        self.correlation_layer = Correlation_Layer(self.view_size)

        base = vgg16(pretrained=False).features
        self.corr_encoder = base[:15].to(self.device[0])

        self.weight_pred = nn.Sequential(nn.Conv2d(3600, 64, 1), nn.ReLU(),
                                             nn.Conv2d(64, 64, 3, dilation=1, padding='same'), nn.ReLU(),
                                             nn.Conv2d(64, 32, 3, dilation=1, padding='same'), nn.ReLU(),
                                             nn.Conv2d(32, 1, 1, dilation=1, padding='same'), nn.ReLU(),
                                             nn.Flatten(),
                                             nn.Linear(45*80, 64),
                                             nn.Linear(64, 8)).to(self.device[0])
        self.init_parameters()

    def init_parameters(self):
        # for m in self.corr_encoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # m.weight.data.fill_(0.0001)
        #         # m.bias.data.zero_()
        #         nn.init.normal_(m.weight, std=0.01)
        x = 0
        for m in self.weight_pred.modules():
            if isinstance(m, nn.Linear):
                x += 1
                if x > 1:
                    m.weight.data.zero_()
                    m.bias.data.zero_()
                    m.bias.data[0], m.bias.data[4] = torch.tensor([1.0]), torch.tensor([1.0])
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
        # self.weight_pred_final = nn.Sequential(nn.Linear(64, 8)).to(self.device[0])
        # self.weight_pred_final.weight.data.fill_(0.0)
        # self.weight_pred_final.weight.data[:, [0, 4]] = torch.tensor([[1, 0], [1, 0]])



    def forward(self, imgs, train=True):
        B, N, C, H, W = imgs.shape
        imgs = torch.reshape(imgs, (B*N, C, H, W))

        corr_feature = self.corr_encoder(imgs.to(self.device[0]))
        view_corr = self.correlation_layer(corr_feature)

        homography = self.weight_pred(view_corr)
        ones = torch.ones((homography.shape[0], 1)).to(homography.device)
        homography = torch.cat([homography, ones], dim=-1)
        homography = homography.reshape((B, -1, 3, 3))

        return homography


    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret

def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.loss.gaussian_mse import GaussianMSE
    from lb_utils import _transform
    import os
    from PIL import Image
    from multiview_detector.utils.image_utils import add_heatmap_to_image
    from multiview_detector.utils.image_utils import img_color_denormalize

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])
    dataset = frameDataset(Wildtrack(os.path.expanduser('/mnt/d/common/Datasets/Wildtrack')), transform=train_trans,
                           train=True)
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4, pin_memory=False)

    model = PerspTransDetector_homo(dataset)

    weights_name = "/mnt/d/common/DJ/Calibration-free/logs/wildtrack_frame/homo/2024-11-05_12-20-51/models/MultiviewDetector_epoch24.pth"
    pretrain_model = torch.load(weights_name)

    model_para = model.state_dict()
    pretrain_para = pretrain_model['model']
    # useful_para = {k.replace('base_pt', 'corr_encoder'): v for k, v in pretrain_para.items() if k.replace('base_pt', 'corr_encoder') in model_para}
    useful_para = {k: v for k, v in pretrain_para.items() if k in model_para}

    model_para.update(useful_para)
    model.load_state_dict(model_para)

    # img_path_name = r'/mnt/data/Datasets/CVCS/train/scene_45/0/jpgs/95.jpg'
    # img = cv2.imread(img_path_name)
    # img = img[:, :, (2, 1, 0)]  # BGR -> RGB
    # img = img.astype('float32')
    #
    # img = img / 255.0
    # img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    # img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    # img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    # # downsample
    # img = cv2.resize(img, (640, 360))
    # img = torch.asarray(img).permute(2, 0, 1).unsqueeze(0)
    # img = img.repeat(2, 1, 1, 1)
    for batch_idx, (data, imgs_gt, map_gt, homography) in enumerate(dataloader):
        B, N, C, H, W = data.shape
        homography = homography.reshape((B, N * (N - 1), 3, 3))

        homo_res = model(data, train=False)
        loss = 0
    pass

if __name__ == '__main__':
    test()