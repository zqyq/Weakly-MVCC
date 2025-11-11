import os

from multiview_detector.loss.Eval_metric import GAME_metric

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T

from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE

from multiview_detector.models.persp_trans_detector_max import PerspTransDetector_max as PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant

from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])
    if 'wildtrack' in args.dataset:
        # data_path = os.path.expanduser('~/Data/Wildtrack')
        data_path = os.path.expanduser('/root/home/Daijie/Data/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('~/Data/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, args.rate, train=True, transform=train_trans, grid_reduce=4)
    test_set = frameDataset(base, 1., train=False, transform=train_trans, grid_reduce=4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch)
    else:
        raise Exception('no support for this variant')

    # # load the model:
    # # weights_name = 'logs/wildtrack_frame/default/2021-11-01_10-20-03/MultiviewDetector.pth'
    # weights_name = 'logs/wildtrack_frame/default/2021-10-20_19-46-36/MultiviewDetector.pth'

    # weights_name = 'logs/wildtrack_frame/default/2021-11-05_21-31-59/models/MultiviewDetector_epoch3.pth'
    # weights_name = 'logs/wildtrack_frame/default/2021-11-06_19-41-06/models/MultiviewDetector_epoch12.pth'
    # model.load_state_dict(torch.load(weights_name, map_location=torch.device('cuda:0')))

    for param in model.base_pt1.parameters():
        param.requires_grad = False
    for param in model.base_pt2.parameters():
        param.requires_grad = False

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)
    # loss
    criterion = GaussianMSE().cuda()

    pretrain = True
    if pretrain:
        weights_name = "/root/home/Daijie/Semi_2D_Counting/logs/wildtrack_frame/0.1/2023-04-20_01-49-41/models/MultiviewDetector_epoch13.pth"
        pretrain_model = torch.load(weights_name)

        model_para = model.state_dict()
        pretrain_para = pretrain_model['model']
        useful_para = {k: v for k, v in pretrain_para.items() if k in model_para}
        model_para.update(useful_para)
        model.load_state_dict(model_para)

    # learn
    if args.resume is None:
        # print('Testing...')
        # trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)
        GAMES_2D = [0,0,0,0]
        for batch_idx, (data, imgs_gt, camera_paras, wld_map_paras, hw_random, map_gt, map_counts) in enumerate(test_loader):
            if map_gt.flatten().max == 0:  # zq
                continue

            with torch.no_grad():
                map_res, imgs_res = model(data, camera_paras, wld_map_paras, hw_random)
                GAMES_2D_A = (torch.clamp(map_res.cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2, 0).numpy() / 100
                GAMES_2D_B = (torch.clamp(map_gt.cpu().detach(), min=0)).permute(1, 2, 0).numpy() / 100

                for i in range(3):
                    GAMES_2D[i] += GAME_metric(GAMES_2D_A, GAMES_2D_B, i)
        print(GAMES_2D[0].item() / (len(test_loader) + 1))
        print(GAMES_2D[1].item() / (len(test_loader) + 1))
        print(GAMES_2D[2].item() / (len(test_loader) + 1))





if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--rate', type=float, default=0.1)
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='vgg11', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    args = parser.parse_args()

    main(args)
