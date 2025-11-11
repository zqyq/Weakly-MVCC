import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from lb_utils import _transform
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.loss.Eval_metric import GAME_metric


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.mae_loss = torch.nn.L1Loss(reduction='sum')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        losses1, losses2 = 0, 0
        mae, nae, mse = 0, 0, 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        xx = 0
        t_forward = 0
        t_backward = 0

        for batch_idx, (data, imgs_gt, map_gt, homography) in enumerate(data_loader):
            B, N, C, H, W = data.shape
            homography = homography.reshape((B, N * (N - 1), 3, 3))

            optimizer.zero_grad()

            homo_res = self.model(data, train=False)
            t_f = time.time()
            t_forward += t_f - t_b

            loss1, loss2 = 0, 0
            for b in range(B):
                for i in range(homo_res.shape[1]):
                    feature_mask = torch.ones((1, 360, 640, 1)).to(homo_res.device).requires_grad_(True)
                    # id = int(i / 2)
                    # feature_mask = data[id:id+1].to(homo_res.device)
                    trans_mask1 = _transform(feature_mask, homo_res[b][i:i+1])
                    trans_mask2 = _transform(feature_mask, homography[b][i:i+1].float().to(homo_res.device))
                    # loss1 += F.mse_loss(trans_mask1, trans_mask2)  / (homo_res.shape[0])
                    loss1 += self.mse_loss(trans_mask1 * 5, trans_mask2 * 5) / (N * (N - 1) * B + 1e-8)
                    loss2 += self.mae_loss(homo_res[b][i:i+1], homography[b][i:i+1].float().to(homo_res.device)) / (N * (N - 1) * B + 1e-8)
                    # loss2 += torch.sum(torch.abs(homo_res[b][i:i+1] - homography[b][i:i+1].float().to(homo_res.device))) / (N * (N - 1) * B + 1e-8)
            if loss2 > 1e8:
                # print(homo_res)
                # print("-------------------")
                # print(homography)
                # print(str(batch_idx) + ': loss too large!')
                xx += 1
                if xx > 20:
                    continue
                continue
            loss = (1 / 3) * loss1 + 0.5 * loss2

            loss.backward()
            optimizer.step()
            losses += loss.item()
            losses1 += loss1.item()
            losses2 += loss2.item()

            # for name, parms in self.model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #           ' -->grad_value:', torch.mean(parms.grad))

            t_b = time.time()
            t_backward += t_b - t_f

            if (batch_idx + 1) % 200 == 0:
                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="output")
                subplt1 = fig.add_subplot(122, title="target")
                subplt0.imshow(trans_mask1[0, 0, :, :].cpu().detach().numpy())
                subplt1.imshow(trans_mask2[0, 0, :, :].cpu().detach().numpy())
                plt.savefig(os.path.join(self.logdir, f'train_img_{str(batch_idx + 1)}.jpg'))
                plt.close(fig)


            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, lr: {:.6f}, '
                      'Loss1: {:.6f}, Loss2: {:.6f}, Time: {:.1f} (f{:.3f}+b{:.3f})'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), current_lr, losses1 / (batch_idx + 1), losses2 / (batch_idx + 1), t_epoch, t_forward / batch_idx,
                                            t_backward / batch_idx))
                # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.1f} (f{:.3f}+b{:.3f})'.format(
                #     epoch, (batch_idx + 1), losses / (batch_idx + 1), t_epoch, t_forward / batch_idx, t_backward / batch_idx))
                pass

        if cyclic_scheduler is not None:
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
                cyclic_scheduler.step()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), t_epoch))
        # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.3f}'.format(
        #     epoch, len(data_loader), losses / len(data_loader), t_epoch))

        return losses / len(data_loader), 0

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        mae, nae, mse = 0, 0, 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        all_gt_list = []
        GAMES_2D = [0, 0, 0, 0]

        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None

        # for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
        # img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map
        for batch_idx, (data, imgs_gt, map_gt, homography) in enumerate(
                data_loader):
            if map_gt.flatten().max == 0:  # zq
                continue

            with torch.no_grad():
                B, N, C, H, W = data.shape
                homography = homography.reshape((B, N * (N - 1), 3, 3))

                homo_res = self.model(data, train=False)

            loss = F.mse_loss(homo_res, homography.float().to(homo_res.device)) / (homo_res.shape[0])


            # GAMES_2D_A = (torch.clamp(map_res[:, 0:1].cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2,
            #                                                                                          0).numpy() / 100
            # GAMES_2D_B = (torch.clamp(map_gt[:, 0:1].cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2,
            #                                                                                         0).numpy() / 100
            # for i in range(3):
            #     GAMES_2D[i] += GAME_metric(GAMES_2D_A, GAMES_2D_B, i)

            losses += loss.item()
            if (batch_idx + 1) % 100 == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.1f}'.format(
                    1, (batch_idx + 1), losses / (batch_idx + 1), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0

        print('Test, Loss: {:.6f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), t_epoch))
        # print('Test, Loss: {:.6f}, Time: {:.3f}'.format(losses / (len(data_loader) + 1), t_epoch))

        return losses / len(data_loader), 0, 0


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from lb_utils import _transform
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.loss.Eval_metric import GAME_metric


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.mae_loss = torch.nn.L1Loss(reduction='sum')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        losses1, losses2 = 0, 0
        mae, nae, mse = 0, 0, 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        xx = 0
        t_forward = 0
        t_backward = 0

        for batch_idx, (data, imgs_gt, map_gt, homography) in enumerate(data_loader):
            B, N, C, H, W = data.shape
            homography = homography.reshape((B, N * (N - 1), 3, 3))

            optimizer.zero_grad()

            homo_res = self.model(data, train=False)
            t_f = time.time()
            t_forward += t_f - t_b

            loss1, loss2 = 0, 0
            for b in range(B):
                for i in range(homo_res.shape[1]):
                    feature_mask = torch.ones((1, 360, 640, 1)).to(homo_res.device).requires_grad_(True)
                    # id = int(i / 2)
                    # feature_mask = data[id:id+1].to(homo_res.device)
                    trans_mask1 = _transform(feature_mask, homo_res[b][i:i+1])
                    trans_mask2 = _transform(feature_mask, homography[b][i:i+1].float().to(homo_res.device))
                    # loss1 += F.mse_loss(trans_mask1, trans_mask2)  / (homo_res.shape[0])
                    loss1 += self.mse_loss(trans_mask1 * 5, trans_mask2 * 5) / (N * (N - 1) * B + 1e-8)
                    loss2 += self.mae_loss(homo_res[b][i:i+1], homography[b][i:i+1].float().to(homo_res.device)) / (N * (N - 1) * B + 1e-8)
                    # loss2 += torch.sum(torch.abs(homo_res[b][i:i+1] - homography[b][i:i+1].float().to(homo_res.device))) / (N * (N - 1) * B + 1e-8)
            if loss2 > 1e8:
                # print(homo_res)
                # print("-------------------")
                # print(homography)
                # print(str(batch_idx) + ': loss too large!')
                xx += 1
                if xx > 20:
                    continue
                continue
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            losses += loss.item()
            losses1 += loss1.item()
            losses2 += loss2.item()

            # for name, parms in self.model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #           ' -->grad_value:', torch.mean(parms.grad))

            t_b = time.time()
            t_backward += t_b - t_f

            if (batch_idx + 1) % 200 == 0:
                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="output")
                subplt1 = fig.add_subplot(122, title="target")
                subplt0.imshow(trans_mask1[0, 0, :, :].cpu().detach().numpy())
                subplt1.imshow(trans_mask2[0, 0, :, :].cpu().detach().numpy())
                plt.savefig(os.path.join(self.logdir, f'train_img_{str(batch_idx + 1)}.jpg'))
                plt.close(fig)


            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, lr: {:.6f}, '
                      'Loss1: {:.6f}, Loss2: {:.6f}, Time: {:.1f} (f{:.3f}+b{:.3f})'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), current_lr, losses1 / (batch_idx + 1), losses2 / (batch_idx + 1), t_epoch, t_forward / batch_idx,
                                            t_backward / batch_idx))
                # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.1f} (f{:.3f}+b{:.3f})'.format(
                #     epoch, (batch_idx + 1), losses / (batch_idx + 1), t_epoch, t_forward / batch_idx, t_backward / batch_idx))
                pass

        if cyclic_scheduler is not None:
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
                cyclic_scheduler.step()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), t_epoch))
        # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.3f}'.format(
        #     epoch, len(data_loader), losses / len(data_loader), t_epoch))

        return losses / len(data_loader), 0

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
        self.model.eval()
        losses = 0
        mae, nae, mse = 0, 0, 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        all_gt_list = []
        GAMES_2D = [0, 0, 0, 0]

        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None

        # for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
        # img_views, single_view_dmaps, camera_paras, wld_map_paras, hw_random, GP_density_map
        for batch_idx, (data, imgs_gt, map_gt, homography) in enumerate(
                data_loader):
            if map_gt.flatten().max == 0:  # zq
                continue

            with torch.no_grad():
                B, N, C, H, W = data.shape
                homography = homography.reshape((B, N * (N - 1), 3, 3))

                homo_res = self.model(data, train=False)

            loss = F.mse_loss(homo_res, homography.float().to(homo_res.device)) / (homo_res.shape[0])


            # GAMES_2D_A = (torch.clamp(map_res[:, 0:1].cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2,
            #                                                                                          0).numpy() / 100
            # GAMES_2D_B = (torch.clamp(map_gt[:, 0:1].cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2,
            #                                                                                         0).numpy() / 100
            # for i in range(3):
            #     GAMES_2D[i] += GAME_metric(GAMES_2D_A, GAMES_2D_B, i)

            losses += loss.item()
            if (batch_idx + 1) % 100 == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.1f}'.format(
                    1, (batch_idx + 1), losses / (batch_idx + 1), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0

        print('Test, Loss: {:.6f}, \tTime: {:.3f}'.format(
            losses / (len(data_loader) + 1), t_epoch))
        # print('Test, Loss: {:.6f}, Time: {:.3f}'.format(losses / (len(data_loader) + 1), t_epoch))

        return losses / len(data_loader), 0, 0


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
