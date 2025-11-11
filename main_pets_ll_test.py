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
        args.num_workers = 1
        torch.autograd.set_detect_anomaly(True)
    else:
        print('Not debugging...')
        is_debug = False

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])

    # if 'wildtrack' in args.dataset:
    #     # data_path = os.path.expanduser('~/Data/Wildtrack')
    #     data_path = os.path.expanduser('/mnt/d/data/Wildtrack')
    #     base = Wildtrack(data_path)
    # elif 'multiviewx' in args.dataset:
    #     data_path = os.path.expanduser(args.dataset_dir + '/MultiviewX')
    #     base = MultiviewX(data_path)
    # else:
    #     raise Exception('must choose from [wildtrack, multiviewx]')

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

    # for param in model.base_pt2.parameters():
    #     param.requires_grad = False
    # for param in model.base_pt.parameters():
    #     param.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
    #                                                 epochs=args.epochs)
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
        # first use lr with 0.01 to train, then retrain the network 100Epoch with init weight Epoch28 and loss img
        # then train network with map loss and 0.5 img loss 66Epoch
        # retrain map gt with trained 0.5 img and map 100Epoch
        # img_classifer
        if args.multi_view_train:
            weights_name = None

            if True:
            # if not is_debug:

                assert args.ckpt is not None

                weights_name = args.ckpt
                pretrain_model = torch.load(weights_name)
                pretrain_para = pretrain_model['model']

                model_para = model.state_dict()
                useful_para1 = {k: v for k, v in pretrain_para.items() if k in model_para}
                model_para.update(useful_para1)
                model.load_state_dict(model_para)
                print('load single view prediction weight: ', weights_name)

                if not args.multi_view_train_fromScratch:
                    # weight_pred_name = "/mnt/d/DJ/CF/ pretrain_model/pets/01-11279444-MAE4.99-better.pth"
                    weight_pred_name = '/mnt/d/DJ/CF/logs/pets_frame/pets/2025-05-27_02-02-19/models/_epoch2_MAE2.9224962534394656.pth'
                    pretrain_model = torch.load(weight_pred_name)
                    pretrain_para = pretrain_model['model']

                    model_para = model.state_dict()
                    useful_para2 = {k: v for k, v in pretrain_para.items() if k in model_para and k.split('.')[0] in ['distence_extractor', 'match_CNN', 'corr_encoder', 'confidence_decoder', 'weight_pred']}
                    useful_para = dict(list(useful_para1.items()) + list(useful_para2.items()))
                    model_para.update(useful_para)
                    model.load_state_dict(model_para)
                    print('load weight prediction weight: ', weight_pred_name)

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

                # weights_name = "logs/pets_frame/pets/debug/2025-05-20_17-24-59/models/Ep_100_mae_14.842425915574623_mse_309.5074898468078.pth"
                # weights_name = "logs/multiviewx_frame/1/debug/2023-11-06_14-37-48/models/MultiviewDetector_epoch100.pth"
                pretrain_model = torch.load(weights_name)

                model_para = model.state_dict()
                pretrain_para = pretrain_model
                # optimizer.load_state_dict(pretrain_model['optim'])
                # optimizer.param_groups[0]['lr'] *= 10
                # optimizer.state_dict()['weight_decay'] = args.weight_decay
                # scheduler.load_state_dict(pretrain_model['scheduler'])
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

            # x_epoch.append(epoch)
            # train_loss_s.append(train_loss)
            # train_prec_s.append(train_prec)
            # test_loss_s.append(test_loss)
            # test_prec_s.append(test_prec)
            # test_moda_s.append(mse)
            # draw_curve_me(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s,
            #            test_loss_s)
            # draw_curve_me(os.path.join(logdir, 'MAE_curve.jpg'), x_epoch, train_prec_s,
            #               test_prec_s)
            # save
            # torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
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

# nohup python main_pets_ll.py --cuda_order 5 --logdir ../CVCS_FewSample_240901_latest/logs/cvcs/2024-11-07_07-18-38 --lbtask_method exp2/origin_latest_remove0json_AvgDistM2 --mode test --model fpn --dataset_type CVCS_for_test_by_selected_view --maxViews 5 --seed 14 --selected_scene_number -1 --name CoverRate_MVMSOrigin_order4 > logs/CoverRate_MVMSOrigin_order4.log 2>&1 &
# nohup python main_pets_ll.py --cuda_order 3 --name single_view_with_rankLossRatio10_lre-7 --lr 5e-7 --rank_loss_ratio 5 > logs/single_view_with_rankLossRatio5_lr5e-7.log 2>&1 &

# nohup python main_pets_ll.py --cuda_order 0 --mode test --name single_view_with_rankLossRatio10_lr5e --ckpt logs/pets_frame/pets/debug/2025-05-20_17-24-59/models/Ep_102_mae_14.433891155766178_mse_291.4002037023859.pth > logs/test_single_view_with_rankLossRatio10_lr5e.log 2>&1 &
# nohup python main_pets_ll.py --cuda_order 5 --mode test --name single_view_with_rankLossRatio5_lr1e-7 --ckpt logs/pets_frame/pets/debug/2025-05-20_17-26-50/models/Ep_200_mae_14.718304309351744_mse_303.91469438137847.pth > logs/test_single_view_with_rankLossRatio5_lr1e-7.log 2>&1 &

# nohup python main_pets_ll.py --cuda_order 6 --mode test --name single_view_with_rankLossRatio5_lr5e-7 --ckpt logs/pets_frame/pets/debug/2025-05-20_17-27-18/models/Ep_358_mae_14.645704966396623_mse_300.93212584664366.pth > logs/test_single_view_with_rankLossRatio5_lr5e-7.log 2>&1 &
# nohup python main_pets_ll.py --cuda_order 7 --mode test --name single_view_with_rankLossRatio10_lr1e-7 --ckpt logs/pets_frame/pets/debug/2025-05-20_17-09-23/models/Ep_101_mae_14.671483656773473_mse_302.3301469037394.pth > logs/test_single_view_with_rankLossRatio10_lr1e-7.log 2>&1 &
# nohup python main_pets_ll.py --cuda_order 7 --mode test --name single_view_with_rankLossRatio5_lr5e-7_onlySingleView --ckpt logs/pets_frame/pets/debug/2025-05-21_03-25-34/models/Ep_500_mae_14.624003819548513_mse_300.1351882599188.pth > logs/test_single_view_with_rankLossRatio5_lr5e-7_onlySingleView.log 2>&1 &

if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--cuda_order', type=str, default='7')
    parser.add_argument('--rank_loss_ratio', type=float, default=0)
    parser.add_argument('--onlyTrainSingleView', action='store_true')
    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='../calibrationFree/logs/pets_frame/pets/debug/2025-05-24_08-03-23/models/Ep_25_mae_3.863309015411338_mse_26.575469027983267.pth')
    parser.add_argument('--density_scale', type=int, default=100)
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--multi_view_train', action='store_true', default=False)
    parser.add_argument('--multi_view_train_fromScratch', action='store_true', default=True)
    parser.add_argument('--multi_view_loss1_ratio', type=float, default=0.0001)
    parser.add_argument('--multi_view_loss2_ratio', type=float, default=0)

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
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 0.1)')
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

