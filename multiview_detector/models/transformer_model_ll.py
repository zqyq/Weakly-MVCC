import os
import numpy as np
import torch
import torch.nn as nn
from multiview_detector.models.trans_img_feat import TransformerWorldFeat
from multiview_detector.models.VGG_ll import vgg19
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector_img(nn.Module):
    def __init__(self, dataset, args='vgg11', bottleneck_dim=128):  # vgg11 resnet18
        super().__init__()

        # end--------------------
        self.batch_size = dataset.batch_size
        self.view_size = dataset.view_size
        self.patch_num = dataset.patch_num  # it is used to control the number of sample block in single image

        # output_size = [96, 128]
        output_size = [72, 96]

        self.output_size = output_size
        # self.imgsize = [90, 160]  #
        self.imgsize = output_size # 384, 288
        self.reducedgrid_shape = output_size

        self.device = ["cuda:0", "cuda:1"]
        self.num_cam = dataset.num_cam

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        out_channel = 512
        # base = vgg16().features
        self.base_pt = vgg19().to(self.device[0])

        if bottleneck_dim:
            self.bottleneck = nn.Sequential(nn.Conv2d(out_channel, bottleneck_dim, 1), nn.Dropout2d(0.0)).to(self.device[0])
            out_channel = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()

        self.world_feat = TransformerWorldFeat(1, self.imgsize, out_channel).to(self.device[0])

        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 3, padding='same'), nn.ReLU(),
                                            nn.Conv2d(64, 32, 3, padding='same'), nn.ReLU(),
                                            nn.Conv2d(32, 1, 1)).to(self.device[0])  # 7

    def init_weights_depth(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_weights_nondepth(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.normal_(layer.weight, std=0.01)

    def forward(self, imgs, train=True):
        B, N, C, H, W = imgs.size()
        # assert N == self.num_cam

        imgs = imgs.view(-1, C, H, W)
        img_feature = self.base_pt(imgs.float().to(self.device[0]))

        # channel reduction
        img_feature = self.bottleneck(img_feature).unsqueeze(1)

        feat = self.world_feat(img_feature)
        img_res = self.img_classifier(feat)
        # img_res = torch.where(img_res<0, -img_res, img_res)

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