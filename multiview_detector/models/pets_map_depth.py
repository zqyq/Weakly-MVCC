import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18
from multiview_detector.models.correlation_layer import Correlation_Layer_noNorm as Correlation_Layer
from lb_utils import _transform, _meshgrid, _interpolate
import matplotlib.pyplot as plt

# from multiview_detector.models.spatial_transformer_lowRes_canGradient import spatial_transformation_layer


class PerspTransDetector_max_cvcs(nn.Module):
    def __init__(self, dataset, arch='vgg11'):  # vgg11 resnet18
        super().__init__()

        # end--------------------
        self.batch_size = 1
        self.view_size = 3
        self.patch_num = 1  # it is used to control the number of sample block in single image
        self.output_size = [160, 180]
        self.reducedgrid_shape = [160, 180]

        self.device = ["cuda:0", "cuda:0"]
        self.num_cam = 3

        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        out_channel = 256

        self.correlation_layer = Correlation_Layer(self.view_size)

        self.base_pt = nn.Sequential(nn.Conv2d(1, 64, 3, padding='same'), nn.ReLU(),
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

        self.corr_encoder = nn.Sequential(nn.Conv2d(1, 64, 3, padding='same'), nn.ReLU(),
                                    nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(),
                                    nn.MaxPool2d(2),  # 3

                                    nn.Conv2d(64, 128, 3, padding='same'), nn.ReLU(),
                                    nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(),
                                    nn.MaxPool2d(2),  # 3

                                    nn.Conv2d(128, 256, 3, padding='same'), nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, padding='same'), nn.ReLU()).to(self.device[0])

        self.weight_pred = nn.Sequential(nn.Conv2d(1728, 64, 3, padding='same'), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, dilation=1, padding='same'), nn.ReLU(),
                                         nn.Conv2d(64, 32, 3, dilation=1, padding='same'), nn.ReLU(),
                                         nn.Conv2d(32, 1, 1, dilation=1, padding='same'), nn.ReLU(),
                                         nn.Flatten(),
                                         nn.Linear(1728, 64),
                                         nn.Linear(64, 8)).to(self.device[0])

        self.match_CNN = nn.Sequential(nn.Conv2d(512, 128, 3, stride=1, padding='same'), nn.ReLU(),
                                       nn.Conv2d(128, 64, 3, stride=1, padding='same'), nn.ReLU(),
                                       nn.Conv2d(64, 1, 1, stride=1, padding='same'), nn.Sigmoid()).to(self.device[0])

        self.distence_extractor = nn.Sequential(nn.Conv2d(1, 128, 3, dilation=1, padding='same'), nn.ReLU(),
                                                nn.Conv2d(128, 64, 3, dilation=1, padding='same'), nn.ReLU()).to(self.device[0])

        self.confidence_decoder = nn.Sequential(nn.Conv2d(320, 128, 3, dilation=1, padding='same'), nn.ReLU(),
                                                nn.Conv2d(128, 64, 3, dilation=1, padding='same'), nn.ReLU(),
                                                nn.Conv2d(64, 1, 1, dilation=1, padding='same'),
                                                nn.Sigmoid()).to(self.device[0])

        self.init_parameters()
    def init_weights_depth(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_parameters(self):
        for m in self.match_CNN.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
        for m in self.distence_extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
        for m in self.confidence_decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, imgs, distmap, train=True):
        B, N, C, H, W = imgs.shape

        img_feature = self.base_pt(imgs[0].to(self.device[0]))
        img_res = self.img_classifier(img_feature)

        corr_feature0 = self.corr_encoder(imgs[0].to(self.device[0]))
        corr_feature = nn.MaxPool2d(2)(corr_feature0)
        view_corr = self.correlation_layer(corr_feature)

        homography = self.weight_pred(view_corr)
        ones = torch.ones((homography.shape[0], 1)).to(homography.device)
        homography = torch.cat([homography, ones], dim=-1)
        homography = homography.reshape((N, N-1, -1))

        w = []
        img_feature0 = corr_feature0.permute(0, 2, 3, 1)
        for i in range(N):
            feature_i = img_feature0[i:i+1]
            feature_i = feature_i.repeat(N-1, 1, 1, 1)

            feature_others = torch.cat([img_feature0[:i], img_feature0[i+1:]], dim=0)
            matrix_i = homography[i:i+1]
            matrix_i = matrix_i.reshape((N-1, 3, 3))

            output_i = _transform(feature_others, matrix_i).permute(0, 2, 3, 1)
            union_feature = torch.cat((feature_i, output_i), dim=-1).permute(0, 3, 1, 2)

            Mij = self.match_CNN(union_feature)
            w.append(Mij)

        w = torch.cat(w, dim=1).permute(1, 0, 2, 3)
        # w = torch.reshape(w, (N, N-1, img_feature.shape[1], img_feature.shape[2]))
        # w = torch.sum(w, dim=1, keepdim=True)

        # weight = 1 / (1 + w)
        dist_feature = self.distence_extractor(distmap[0].float().to(self.device[0]))
        # dist_feature *= corr_feature.max()
        extra_feature = torch.cat([corr_feature0, dist_feature], dim=1)
        dist_score = self.confidence_decoder(extra_feature)
        # dist_score = (dist_score - torch.min(dist_score, dim=0, keepdim=True)[0]) / (torch.max(dist_score, dim=0, keepdim=True)[0] - torch.min(dist_score, dim=0, keepdim=True)[0] + 1e-5)
        # print(str(dist_feature.max()) + " " + str(dist_score.max()))

        feature_cat = []
        for i in range(N):
            dist_others = torch.cat([dist_score[:i], dist_score[i+1:]], dim=0).permute(0, 2, 3, 1)
            # dist_others = dist_others.detach()
            matrix_i = homography[i:i+1]
            matrix_i = matrix_i.reshape((N - 1, 3, 3))

            dist_output_i = _transform(dist_others, matrix_i)
            feature_cat.append(dist_output_i)

        dist_score_ij = torch.cat(feature_cat, dim=1).permute(1, 0, 2, 3)
        # dist_score_ij = torch.where(dist_score_ij <= 0, 1e-8, dist_score_ij)

        w_d_ij = torch.sum(w*dist_score_ij, dim=1, keepdim=True)
        weight =  dist_score /(w_d_ij + dist_score + 1e-8)
        # weight = F.interpolate(weight, (95, 169), mode='bilinear')


        x_output = torch.mul(img_res, weight)
        x_output = torch.reshape(x_output, (B, -1))
        x_output_num = torch.sum(x_output, dim=-1, keepdim=True) / 1000

        return x_output_num, w, dist_score



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
    import torch.optim as optim
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
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-5)

    weight_name = "/mnt/d/DJ/CF/logs/wildtrack_frame/final/dist/2024-12-02_15-15-08/models/MultiviewDetector_epoch9.pth"
    pretrain_model = torch.load(weight_name)
    pretrain_para = pretrain_model['model']

    model_para = model.state_dict()
    useful_para = {k: v for k, v in pretrain_para.items() if k in model_para}
    model_para.update(useful_para)
    model.load_state_dict(model_para)

    mse_loss = nn.MSELoss()

    for batch_idx, (data, imgs_gt, map_gt, homography, depthMap) in enumerate(
            dataloader):
        # B, N, C, H, W = data.shape
        # depthMap, homography = depthMap[0], homography[0].reshape(N, N-1, 3, 3)
        # optimizer.zero_grad()
        res, _ = model(data, depthMap)

        loss = mse_loss(res.sum(), map_gt.sum().to(res.device))
        loss.backward()
        # optimizer.step()

        for name, parms in model.named_parameters():
            # if parms.gard > 10000:
            #     print(1)
            if parms.grad is None or parms.data is None:
                pass
                # if parms.grad is None and parms.data is not None:
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                #           torch.mean(parms.data))
                # elif parms.data is None and parms.grad is not None:
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                #           ' -->grad_value:', torch.mean(parms.grad))
                # else:
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
            else:
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                      torch.mean(parms.data),
                      ' -->grad_value:', torch.mean(parms.grad))

    pass

if __name__ == '__main__':
    test()