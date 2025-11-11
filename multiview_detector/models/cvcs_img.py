import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector_img(nn.Module):
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
        # base = vgg16().features
        self.base_pt = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(),
                                     nn.MaxPool2d(2),  # 3

                                     nn.Conv2d(64, 128, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                     nn.MaxPool2d(2),  # 3

                                     nn.Conv2d(128, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU()).to(self.device[0])
        # self.base_pt = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'), nn.ReLU(),  # conv1
        #                              nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(),  # conv2
        #                              nn.MaxPool2d(2),
        #                              nn.Conv2d(64, 128, 3, padding='same'), nn.ReLU(),  # conv3
        #                              nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),  # conv4
        #                              nn.MaxPool2d(2),
        #                              nn.Conv2d(128, 256, 3, padding='same'), nn.ReLU(),  # conv5
        #                              nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),  # conv6
        #                              nn.Conv2d(256, out_channel, 3, padding='same')).to(
        #     self.device[0])  # conv7
        # self.base_pt.apply(self.init_weights_nondepth)

        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding='same'), nn.ReLU(),  # 1
                                            nn.Conv2d(512, 512, 3, padding='same'), nn.ReLU(),  # 2
                                            nn.Conv2d(512, 512, 3, padding='same'), nn.ReLU(),  # 3

                                            nn.Conv2d(512, 512, 3, dilation=2, padding='same'), nn.ReLU(),  # 3
                                            nn.Conv2d(512, 512, 3, dilation=2, padding='same'), nn.ReLU(),  # 3
                                            nn.Conv2d(512, 512, 3, dilation=2, padding='same'), nn.ReLU(),  # 3

                                            nn.Conv2d(512, 256, 3, dilation=2, padding='same'), nn.ReLU(),  # 4
                                            nn.Conv2d(256, 128, 3, dilation=2, padding='same'), nn.ReLU(),  # 5
                                            nn.Conv2d(128, 64, 3, dilation=2, padding='same'), nn.ReLU(),  # 6
                                            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])  # 7

    def init_weights_depth(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_weights_nondepth(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight, std=0.01)

    def forward(self, imgs, train=True):
        img_feature = self.base_pt(imgs.to(self.device[0]))
        img_res = self.img_classifier(img_feature)

        return img_res


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
    from multiview_detector.utils.meters import AverageMeter
    import os
    from PIL import Image
    from multiview_detector.utils.image_utils import add_heatmap_to_image
    from multiview_detector.utils.image_utils import img_color_denormalize

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])
    dataset = frameDataset(Wildtrack(os.path.expanduser('/mnt/data/Datasets/Wildtrack')), transform=train_trans,
                           train=False)
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4, pin_memory=False)

    model = PerspTransDetector_img(dataset)
    img = torch.ones((1, 3, 360, 640))
    x = model(img)
    pass

if __name__ == '__main__':
    test()