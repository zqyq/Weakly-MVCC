import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from multiview_detector.models.VGG import vgg19
from multiview_detector.models.resnet import resnet18
from multiview_detector.models.trans_img_feat import TransformerWorldFeat
from multiview_detector.models.correlation_layer import Correlation_Layer
from lb_utils import _transform
from multiview_detector.models.Conv4D_Layer import Conv4D
import matplotlib.pyplot as plt



# from multiview_detector.models.spatial_transformer_lowRes_canGradient import spatial_transformation_layer


class PerspTransDetector_max_cvcs(nn.Module):
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

        self.base_pt = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(),
                                     nn.MaxPool2d(2),  # 3

                                     nn.Conv2d(64, 128, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                     nn.MaxPool2d(2),  # 3

                                     nn.Conv2d(128, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                     nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU()).to(self.device[0])

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
                                            nn.Conv2d(64, 1, 1, dilation=1, padding='same')).to(self.device[0])  # 7

        self.corr_encoder = nn.Sequential(nn.Conv2d(3, 64, 3, padding='same'), nn.ReLU(),
                                          nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(),
                                          nn.MaxPool2d(2),  # 3

                                          nn.Conv2d(64, 128, 3, padding='same'), nn.ReLU(),
                                          nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                          nn.MaxPool2d(2),  # 3

                                          nn.Conv2d(128, 256, 3, padding='same'), nn.ReLU(),
                                          nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                          nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU()).to(self.device[0])

        self.weight_fusion_pred = nn.Sequential(Conv4D(10, kernel_size=(3, 3, 3, 3), padding='same', activation='relu'),
                                                Conv4D(10, kernel_size=(3, 3, 3, 3), padding='same', activation='relu'),
                                                Conv4D(1, kernel_size=(3, 3, 3, 3), padding='same', activation='relu'))



        # self.init_parameters()
    def init_weights_depth(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_parameters(self):
        for m in self.match_CNN.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
        for m in self.distence_extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
        for m in self.confidence_decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, imgs, distmap=None, train=True):
        B, N, C, H, W = imgs.shape

        img_feature = self.base_pt(imgs[0].to(self.device[0]))
        img_res = self.img_classifier(img_feature)

        corr_feature = self.corr_encoder(imgs[0].to(self.device[0]))
        x12_4D_corr = self.Correlation_4D_Layer2([corr_feature[0:1], corr_feature[1:2]])
        x13_4D_corr = self.Correlation_4D_Layer2([corr_feature[0:1], corr_feature[2:3]])
        x23_4D_corr = self.Correlation_4D_Layer2([corr_feature[1:2], corr_feature[2:3]])

        x_4D_corr = torch.stack([x12_4D_corr, x13_4D_corr, x23_4D_corr], dim=0)
        wij_corr = self.weight_fusion_pred(x_4D_corr)

        b, fs1, fs2, fs3, fs4, ch = wij_corr.shape
        corr4d_B = torch.reshape(wij_corr, (b, fs1*fs2, fs3, fs4))
        corr4d_A = torch.reshape(wij_corr, (b, fs1, fs2, fs3*fs4))

        corr4d_B_max = torch.max(corr4d_B, dim=1, keepdim=True)[0]
        corr4d_A_max = torch.max(corr4d_A, dim=-1, keepdim=True)[0]

        eps = 1e-5
        corr4d_B = corr4d_B / (corr4d_B_max + eps)
        corr4d_A = corr4d_A / (corr4d_A_max + eps)

        # corr4d_B = torch.reshape(corr4d_B, (b, fs1, fs2, fs3, fs4, 1))
        # corr4d_A = torch.reshape(corr4d_A, (b, fs1, fs2, fs3, fs4, 1))
        # corr4d = wij_corr * (corr4d_A * corr4d_B)
        corr4d_A = torch.softmax(corr4d_A, dim=-1)
        corr4d_B = torch.softmax(corr4d_B, dim=-1)

        wij = torch.max(corr4d_A, dim=-1, keepdim=True)[0]
        wji = torch.max(corr4d_B, dim=1, keepdim=True)[1].permute(0, 2, 3, 1)
        w_cat = torch.cat([wij, wji], dim=-1)

        w1 = 1. / (1. + w_cat[0] + w_cat[1])
        w2 = 1. / (1. + w_cat[0] + w_cat[2])
        w3 = 1. / (1. + w_cat[1] + w_cat[2])

        w = torch.cat([w1, w2, w3], dim=-1).permute(0, 3, 1, 2)

        img_res /= 100
        x_output = torch.mul(img_res, w)
        x_output = torch.reshape(x_output, (B, -1))
        x_output_num = torch.sum(x_output, dim=-1, keepdim=True)

        return x_output_num, w, w_cat

    def Correlation_4D_Layer2(self, x):
        feature_B = x[0]
        feature_A = x[1]

        feature_B = torch.max_pool2d(feature_B, kernel_size=2, stride=2)
        feature_A = torch.max_pool2d(feature_A, kernel_size=2, stride=2)

        b, c, h, w = feature_A.shape

        norm_A = torch.pow(torch.sum(torch.pow(feature_A, 2), dim=1, keepdim=True), 0.5)
        norm_B = torch.pow(torch.sum(torch.pow(feature_B, 2), dim=1, keepdim=True), 0.5)
        feature_A_normed = torch.divide(feature_A, norm_A+1e-5)
        feature_B_normed = torch.divide(feature_B, norm_B+1e-5)

        feature_A_flatten = torch.reshape(feature_A_normed, (b, c, h*w))
        feature_B_flatten = torch.reshape(feature_B_normed, (b, h*w, c))

        corr_AB = torch.matmul(feature_B_flatten, feature_A_flatten)
        corr_AB = torch.reshape(corr_AB, (b, h, w, h, w, 1))

        corr4d_B = torch.reshape(corr_AB.copy(), (b, h*w, h, w))
        corr4d_A = torch.reshape(corr_AB.copy(), (b, h, w, h*w))

        corr4d_B_max = torch.max(corr4d_B, dim=1, keepdim=True)[0]
        corr4d_A_max = torch.max(corr4d_A, dim=-1, keepdim=True)[0]

        eps = 1e-5
        corr4d_B = corr4d_B / (corr4d_B_max + eps)
        corr4d_A = corr4d_A / (corr4d_A_max + eps)

        corr4d_A = torch.reshape(corr4d_A, (b, h, w, h, w, 1))
        corr4d_B = torch.reshape(corr4d_B, (b, h, w, h, w, 1))
        corr_AB = corr_AB * (corr4d_A * corr4d_B)

        return corr_AB


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



def processing_layer(x):
    x = x.unsqueeze(0).permute(0, 1, 3, 4, 2)
    x_clip = torch.clamp(x, 0, 1)

    x_clip2 = (1 - x_clip) * 1e8

    x_e8 = x + x_clip2
    x_e8 = torch.log(x_e8)

    x_min = torch.min(x_e8, dim=1, keepdim=True)[0]
    x_min_tile = torch.tile(x_min, (1, 5, 1, 1, 1))

    x_sum = torch.max(x, dim=1, keepdim=True)[0]
    x_sum_clip = torch.clamp(x_sum, 0, 1)
    x_sum_clip2 = 1 - x_sum_clip

    x_dist = -(torch.square(x_e8 - x_min_tile) / (1))
    x_dist2 = torch.exp(x_dist)
    x_dist2_mask = torch.multiply(x_dist2, x_clip)

    x_dist2_mask_sum = torch.sum(x_dist2_mask, dim=1, keepdim=True)
    x_dist2_mask_sum2 = torch.tile(x_dist2_mask_sum + x_sum_clip2, (1, 5, 1, 1, 1))

    x_dist2_mask_sum2_softmax = torch.divide(x_dist2_mask, x_dist2_mask_sum2)
    x_dist2_mask_sum2_softmax_mask = torch.multiply(x_dist2_mask_sum2_softmax, x_clip)

    return x_dist2_mask_sum2_softmax_mask

def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.loss.gaussian_mse import GaussianMSE
    from multiview_detector.utils.meters import AverageMeter
    import os
    from lb_utils import _transform
    from PIL import Image
    from multiview_detector.utils.image_utils import add_heatmap_to_image
    from multiview_detector.utils.image_utils import img_color_denormalize

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])
    dataset = frameDataset(Wildtrack(os.path.expanduser('/mnt/d/data/Wildtrack')), transform=train_trans,
                           train=True)
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4, pin_memory=False)

    model = PerspTransDetector_max_cvcs(dataset)
    weights_name = "/mnt/d/DJ/CF/logs/wildtrack_frame/final/sup_dist/2024-12-03_14-50-48/models/MultiviewDetector_epoch35.pth"
    pretrain_model = torch.load(weights_name)

    model_para = model.state_dict()
    pretrain_para = pretrain_model['model']
    # useful_para = {k.replace('base_pt', 'corr_encoder'): v for k, v in pretrain_para.items() if k.replace('base_pt', 'corr_encoder') in model_para}
    useful_para = {k: v for k, v in pretrain_para.items() if k in model_para}

    model_para.update(useful_para)
    model.load_state_dict(model_para)

    for batch_idx, (data, imgs_gt, map_gt, homography, depthMap) in enumerate(
            dataloader):
        B, N, C, H, W = data.shape
        _, _ = model(data)

    pass

if __name__ == '__main__':
    test()