import os

os.environ['OMP_NUM_THREADS'] = '1'

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

# from multiview_detector.models.cvcs_img import PerspTransDetector_img as PerspTransDetector
from multiview_detector.models.transformer_model_ll import PerspTransDetector_img as PerspTransDetector
from multiview_detector.models.pets_map_depth_ll import PerspTransDetector_max_cvcs
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant

from multiview_detector.utils.logger import Logger
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer_img_ll import PerspectiveTrainer as PerspectiveTrainer
from multiview_detector.trainer_weight_ll import PerspectiveTrainer_weight


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_order

    print("Hello lxb, have a nice result!")
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_order

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Debugging...')
        is_debug = True
        torch.autograd.set_detect_anomaly(True)
    else:
        print('Not debugging...')
        is_debug = False

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])


    data_path = os.path.expanduser('/mnt/d/data/PETS2009/')
    base = Pets_ll(data_path)

    train_set = frameDatasetPets(base, train=True, args=args)

    test_set = frameDatasetPets(base, train=False, args=args)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    print(len(train_loader))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    print(len(test_loader))

    # model
    if args.variant == 'default':
        if args.multi_view_train:
            model = PerspTransDetector_max_cvcs(train_set, args.arch)
        else:
            model = PerspTransDetector(train_set, args.arch)
    else:
        raise Exception('no support for this variant')


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda epoch: 1 / (1 + args.lr_decay * epoch) ** epoch)
    # loss
    criterion = GaussianMSE().cuda()

    # logging
    logdir = f'logs/{args.dataset}_frame/pets/debug/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
        if not args.resume else f'logs/{args.dataset}_frame/{args.variant}/{args.resume}'
    if is_debug:
        logdir = logdir + '_debug'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        # create the models dir:
        os.makedirs(os.path.join(logdir, 'models'), exist_ok=True)

        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    print('logdir: ', logdir)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    if args.multi_view_train:
        trainer = PerspectiveTrainer_weight(model, criterion, logdir, denormalize, args.cls_thres, args.alpha, args=args)
    else:
        trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha, args=args)

    pretrain = True
    if pretrain:
        # img_classifer
        if args.multi_view_train:
            weights_name = None
            if not is_debug:
                assert args.ckpt is not None

                weights_name = args.ckpt
                pretrain_model = torch.load(weights_name)
                pretrain_para = pretrain_model['model']

                model_para = model.state_dict()
                useful_para1 = {k: v for k, v in pretrain_para.items() if k in model_para}
                model_para.update(useful_para1)
                model.load_state_dict(model_para)
                print('load single view prediction weight: ', weights_name)


            # freeze
            for param in model.base_pt.parameters():
                param.requires_grad = False
            for param in model.bottleneck.parameters():
                param.requires_grad = False
            for param in model.world_feat.parameters():
                param.requires_grad = False
            for param in model.img_classifier.parameters():
                param.requires_grad = False
            if not args.multi_view_train_fromScratch:
                for param in model.weight_pred.parameters():
                    param.requires_grad = False
                for param in model.corr_encoder.parameters():
                    param.requires_grad = False
                for param in model.match_CNN.parameters():
                    param.requires_grad = False

        else:
            if args.ckpt is None:
                weights_name = "/mnt/d/DJ/CF/ pretrain_model/DMC_nwpu.pth"

                pretrain_model = torch.load(weights_name)

                model_para = model.state_dict()
                pretrain_para = pretrain_model
                useful_para = {k.replace('features', 'base_pt.features'): v for k, v in pretrain_para.items() if k.replace('features', 'base_pt.features') in model_para}
                model_para.update(useful_para)
                model.load_state_dict(model_para)
            else:
                weights_name = args.ckpt
                pretrain_model = torch.load(weights_name)
                # model_para = model.state_dict()
                model.load_state_dict(pretrain_model['model'])

        print('load weight: ', weights_name)

    isTest = args.mode == 'test'
    best_mae = 1000000
    test_epoch = []

    if isTest:
        print('Testing...')
        testTime = 1
        for i in range(1, testTime + 1):
            print('testTIme: ', i)
            trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True, logdir=logdir)
    else:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler, rank_loss_ratio=args.rank_loss_ratio)
            mae = best_mae
            if epoch % 5 == 0:
                print('Testing...')
                test_loss, test_prec, mse = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                          train_set.gt_fpath, True, epoch=epoch, logdir=logdir)
                mae = test_prec

            if is_debug:
                continue
            if mae < best_mae or epoch % 50 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(logdir, 'models/Ep_' +
                            str(epoch) +
                            '_mae_' +
                            str(mae) +
                            '_mse_' +
                            str(mse) +
                            '.pth'))
                best_mae = mae


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--cuda_order', type=str, default='2')
    parser.add_argument('--rank_loss_ratio', type=float, default=0)
    parser.add_argument('--onlyTrainSingleView', action='store_true')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--density_scale', type=int, default=100)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--multi_view_train', action='store_true')
    parser.add_argument('--multi_view_train_fromScratch', action='store_true')
    parser.add_argument('--multi_view_loss1_ratio', type=float, default=0.1)
    parser.add_argument('--multi_view_loss2_ratio', type=float, default=1000)

    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--rate', type=float, default=1)
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='vgg11', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='pets')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 10)')
    # 0.001
    parser.add_argument('--lr', type=float, default=1e-7, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--lr_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/d/data')
    # origin seed: 1
    # here set 2 for continue img decoder train
    parser.add_argument('--seed', type=int, default=13, help='random seed (default: None)')
    args = parser.parse_args()

    main(args)

