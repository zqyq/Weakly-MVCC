import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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

from multiview_detector.datasets.frameDataset_pets import frameDatasetPets as frameDataset
from multiview_detector.datasets import Wildtrack, MultiviewX, Pets
from multiview_detector.loss.gaussian_mse import GaussianMSE

from multiview_detector.models.pets_map_depth import PerspTransDetector_max_cvcs as PerspTransDetector
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant

from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer_pets import PerspectiveTrainer as PerspectiveTrainer


def main(args):
    print("Here is host, have a nice result!")
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
        data_path = os.path.expanduser('/mnt/d/data/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser(args.dataset_dir + '/MultiviewX')
        base = MultiviewX(data_path)
    elif 'pets' in args.dataset:
        data_path = os.path.expanduser('/mnt/d/data/PETS2009/')
        base = Pets(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base,  train=True)

    test_set = frameDataset(base, train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    print(len(train_loader))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    print(len(test_loader))

    # model
    if args.variant == 'default':
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
    print('main_weight Settings:')
    print(vars(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha)

    pretrain = True
    if pretrain:
        # img_pred_name = "/mnt/d/DJ/CF/ pretrain_model/origion/FeatDist/60-2195.9841.pth"
        # pretrain_model1 = torch.load(img_pred_name)
        #
        # model_para = model.state_dict()
        # pretrain_para1 = pretrain_model1['model']
        #
        # # useful_para1 = {k: v for k, v in pretrain_para1.items() if k in model_para}
        # useful_para1 = {k: v for k, v in pretrain_para1.items() if
        #                 k in model_para and (k.split('.')[0] == 'distence_extractor' or k.split('.')[0] == 'confidence_decoder')}
        # # model_para.update(useful_para1)
        # # model.load_state_dict(model_para)
        # print('load image classifier weight: ', img_pred_name)
        #
        # weight_pred_name = "/mnt/d/DJ/CF/logs/wildtrack_frame/final/sup_dist/2024-12-10_08-58-58/models/MultiviewDetector_epoch93.pth"
        # pretrain_model2 = torch.load(weight_pred_name)
        #
        # pretrain_para2 = pretrain_model2['model']
        # # useful_para2 = {k: v for k, v in pretrain_para2.items() if k in model_para}
        # useful_para2 = {k: v for k, v in pretrain_para2.items() if k in model_para and (k.split('.')[0] != 'distence_extractor' and k.split('.')[0] != 'confidence_decoder')}
        #
        # useful_para = dict(list(useful_para1.items()) + list(useful_para2.items()))
        # model_para.update(useful_para)
        # model.load_state_dict(model_para)
        # print('load weight prediction weight: ', weight_pred_name)
        weight_name = "/mnt/d/DJ/CF/logs/pets_frame/pets/weight/2024-12-31_10-26-58/models/_epoch1_MAE2.948131633254717.pth"
        pretrain_model = torch.load(weight_name)
        pretrain_para = pretrain_model['model']

        model_para = model.state_dict()
        useful_para = {k: v for k, v in pretrain_para.items() if k in model_para and k.split('.')[0] != 'match_CNN'}
        model_para.update(useful_para)
        model.load_state_dict(model_para)
        print('load weight prediction weight: ', weight_name)

    # freeze
    for param in model.base_pt.parameters():
        param.requires_grad = False
    for param in model.img_classifier.parameters():
        param.requires_grad = False
    for param in model.corr_encoder.parameters():
        param.requires_grad = False
    for param in model.weight_pred.parameters():
        param.requires_grad = False
    # for param in model.distence_extractor.parameters():
    #     param.requires_grad = False
    # for param in model.confidence_decoder.parameters():
    #     param.requires_grad = False

    best_mae = 1e10
    isTest = True
    if isTest:
        print('Testing...')
        testTime = 1
        for i in range(1, testTime + 1):
            print('testTIme: ', i)
            trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)
    else:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            # if train_loss > 5:
            #     train_loss = 5.
            print('Testing...')
            test_loss, test_prec, mae = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                      train_set.gt_fpath, True)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            test_moda_s.append(mae)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       test_loss_s, test_prec_s, test_moda_s)
            # save
            # torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
            # if epoch % 2 == 0:
            if mae < best_mae or epoch % 50 == 0 or mae < 3.4:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(logdir, 'models/_epoch' + str(epoch)+ "_MAE" + str(mae) + '.pth'))
                best_mae = mae



if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--rate', type=float, default=1)
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='vgg11', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='pets', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
    # 0.001
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--lr_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/d/data')
    # origin seed: 1
    # here set 2 for continue img decoder train
    parser.add_argument('--seed', type=int, default=24, help='random seed (default: None)')
    args = parser.parse_args()

    main(args)
