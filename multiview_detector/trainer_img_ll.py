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
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.loss.Eval_metric import GAME_metric
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0, args=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.num_cam = model.num_cam
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.mae_loss = torch.nn.L1Loss()
        self.args = args
        self.tb_writer = SummaryWriter(os.path.join(logdir, 'logs'))

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None, rank_loss_ratio=10):
        self.model.train()
        losses = 0
        mae, nae, mse = 0, 0, 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        rank_losses = 0.

        for batch_idx, (data, gp_gt, M_gt, depthMap, imgs_gt, _) in enumerate(
                data_loader):
            data, imgs_gt = data, imgs_gt[0]
            B, V, C, H, W = data.shape

            V = V // 7

            optimizer.zero_grad()

            imgs_res = self.model(data)
            imgs_res = imgs_res.reshape(V, -1, 1, imgs_gt.shape[-2], imgs_gt.shape[-1])
            imgs_gt = imgs_gt.reshape(V, -1, 1, imgs_gt.shape[-2], imgs_gt.shape[-1]).to(imgs_res.device)

            # rows, col = 3, 7

            # fig, axes = plt.subplots(nrows=rows, ncols=col, figsize=(16, 8))  # figsize 控制画布大小
            # # 遍历所有子图并绘图
            # for i in range(rows):
            #     for j in range(col):
            #         ax = axes[i, j]  # 获取当前子图
            #         # ax.imshow(data[0, i * col + j].cpu().permute(1, 2, 0))
            #         ax.imshow(imgs_gt[i, j].cpu().permute(1, 2, 0))
            # # 调整子图间距
            # plt.tight_layout()
            # # plt.savefig(os.path.join(self.logdir, f'imgs_{epoch}_{batch_idx}.png'))
            # plt.show()

            # for i in range(2, 3):
            #     for j in range(col):
            #         plt.imshow(data[0, i * col + j].cpu().permute(1, 2, 0))
            #         plt.axis('off')
            #         plt.tight_layout()
            #         plt.savefig(f'visual/img{j}.jpg', bbox_inches='tight', pad_inches=0)
            #         plt.imshow(imgs_gt[i, j].cpu().permute(1, 2, 0))
            #         plt.axis('off')
            #         plt.tight_layout()
            #         plt.savefig(f'visual/gt{j}.jpg', bbox_inches='tight', pad_inches=0)


            V, MaskN, C, H, W = imgs_res.shape
            t_f = time.time()
            t_forward += t_f - t_b

            loss1 = 0
            for i in range(V):
                # for j in range(3):
                for mask_j in range(MaskN):
                    res = imgs_res[i][mask_j].reshape(1, -1).sum()
                    gt = imgs_gt[i][mask_j].reshape(1, -1).sum()
                    loss1 += self.mae_loss(res, gt.float().to(imgs_res.device)) / 7

            # loss1 /= (MaskN * V)

            Ranking_loss = 0
            for i in range(V):
                # Ranking_loss += (torch.clamp(imgs_res[i][1][0] - imgs_res[i][0][0], min=0).sum()
                #                  + torch.clamp(imgs_res[i][3][0] - imgs_res[i][0][0], min=0).sum() +
                #                  torch.clamp(imgs_res[i][3][0] - imgs_res[i][1][0], min=0).sum())
                # Ranking_loss += (torch.clamp(imgs_res[i][2][0] - imgs_res[i][0][0], min=0).sum()
                #                  + torch.clamp(imgs_res[i][4][0] - imgs_res[i][0][0], min=0).sum() +
                #                  torch.clamp(imgs_res[i][4][0] - imgs_res[i][2][0], min=0).sum())

                Ranking_loss += (torch.clamp(imgs_res[i][5][0] - imgs_res[i][1][0], min=0).sum()
                                 + torch.clamp(imgs_res[i][3][0] - imgs_res[i][1][0], min=0).sum() +
                                 torch.clamp(imgs_res[i][5][0] - imgs_res[i][3][0], min=0).sum())
                Ranking_loss += (torch.clamp(imgs_res[i][6][0] - imgs_res[i][2][0], min=0).sum()
                                 + torch.clamp(imgs_res[i][6][0] - imgs_res[i][4][0], min=0).sum() +
                                 torch.clamp(imgs_res[i][4][0] - imgs_res[i][2][0], min=0).sum())

            # Ranking_loss /= V

            loss = loss1 + rank_loss_ratio * Ranking_loss
            rank_losses += Ranking_loss

            # target_sum = torch.sum(torch.clamp(imgs_gt[0].detach().cpu(), min=0) / 100)
            # # target_all_sum = torch.sum(torch.clamp(map_gt_all.detach().cpu(), min=0) / 10)
            # res_sum = torch.sum(torch.clamp(imgs_res[0, 0].detach().cpu(), min=0) / 100)
            # if target_sum != 0:
            #     MAE = torch.abs(target_sum - res_sum)
            #     NAE = torch.abs(target_sum - res_sum) / (target_sum + 1e-18)
            #     # RMSE
            #     MSE = (target_sum - res_sum) ** 2
            # else:
            #     MAE = res_sum
            #     NAE = 1
            #     # RMSE
            #     MSE = res_sum ** 2
            target_sum = torch.sum(imgs_gt, dim=(2, 3, 4)) / self.args.density_scale
            res_sum = torch.sum(imgs_res.detach(), dim=(2, 3, 4)) / self.args.density_scale

            # target_sum = torch.sum(imgs_gt.detach().cpu()) / 100
            # res_sum = torch.sum(imgs_res.detach().cpu()) / 100

            # if target_sum == 0:
            #     MAE = torch.mean(torch.abs(target_sum - res_sum))
            #     NAE = 1
            #     # RMSE
            #     MSE = torch.mean((target_sum - res_sum) ** 2)
            # else:
            metric_abs = torch.abs(target_sum - res_sum)
            MAE = torch.mean(metric_abs)
            NAE = torch.mean(metric_abs / torch.where(target_sum == 0, metric_abs, target_sum))
            # RMSE
            MSE = torch.mean((target_sum - res_sum) ** 2)

            loss.backward()
            optimizer.step()
            losses += loss.item()
            print(f'batch_idx: {batch_idx + 1}, loss: {loss.item():.6f}, losses: {losses / (batch_idx + 1):.6f}, MAE: {MAE.item():.6f}, NAE: {NAE.item():.6f}, MSE: {MSE.item():.6f}')
            mae += MAE
            nae += NAE
            mse += MSE

            t_b = time.time()
            t_backward += t_b - t_f

            if (batch_idx + 1) % 1000 == 0:
                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="output")
                subplt1 = fig.add_subplot(122, title="target")
                subplt0.imshow(imgs_res[0, 0, :, :].cpu().detach().numpy().squeeze())
                subplt1.imshow(imgs_gt[0, 0, :, :].cpu().detach().numpy().squeeze())
                # plt.show()
                plt.savefig(os.path.join(self.logdir, f'train_img_{str(batch_idx + 1)}.jpg'))
                plt.close(fig)
            #
            #     fig = plt.figure()
            #     subplt0 = fig.add_subplot(121, title="output")
            #     subplt1 = fig.add_subplot(122, title="target")
            #     subplt0.imshow(map_res[0:1, 0:1].cpu().detach().numpy().squeeze())
            #     subplt1.imshow(map_gt[0:1, 0:1].cpu().detach().numpy().squeeze())
            #     plt.savefig(os.path.join(self.logdir, f'train_map_{str(batch_idx + 1)}.jpg'))
            #     plt.close(fig)

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
                      'MAE: {:.3f}, NAE: {:.3f}, RMSE: {:.3f} Time: {:.1f} (f{:.3f}+b{:.3f}), max: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), current_lr, mae.item() / (batch_idx + 1),
                                            nae.item() / (batch_idx + 1),
                                            (mse.item() / (batch_idx + 1)) ** 0.5, t_epoch, t_forward / (batch_idx + 1),
                                            t_backward / (batch_idx + 1), imgs_res.max()))

                # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.1f} (f{:.3f}+b{:.3f})'.format(
                #     epoch, (batch_idx + 1), losses / (batch_idx + 1), t_epoch, t_forward / batch_idx, t_backward / batch_idx))
            pass
            # if (batch_idx + 1) % log_interval == 0:
            #     for name, parms in self.model.named_parameters():
            #         if parms.grad is None or parms.data is None:
            #             if parms.grad is None and parms.data is not None:
            #                 print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #                       torch.mean(parms.data))
            #             elif parms.data is None and parms.grad is not None:
            #                 print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #                       ' -->grad_value:', torch.mean(parms.grad))
            #             else:
            #                 print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
            #         else:
            #             print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #                   torch.mean(parms.data),
            #                   ' -->grad_value:', torch.mean(parms.grad))
        if cyclic_scheduler is not None:
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
                cyclic_scheduler.step()

        len_data_loader = len(data_loader)
        t1 = time.time()
        t_epoch = t1 - t0

        single_view_losses = losses / len_data_loader
        losses = single_view_losses
        rank_losses = rank_losses / len_data_loader
        single_view_MAE = mae.item() / len_data_loader
        single_view_nae = nae.item() / len_data_loader
        single_view_mse = np.sqrt(mse.item() / len_data_loader)

        self.tb_writer.add_scalar('loss/train', losses, epoch)
        self.tb_writer.add_scalar('rank_loss/train', rank_losses, epoch)
        self.tb_writer.add_scalar('single_view_loss/train', losses, epoch)
        self.tb_writer.add_scalar('single_view_mae/train', single_view_MAE, epoch)
        self.tb_writer.add_scalar('single_view_nae/train', single_view_nae, epoch)
        self.tb_writer.add_scalar('single_view_mse/train', single_view_mse, epoch)
        self.tb_writer.add_scalar('time/train', int(t_epoch), epoch)

        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
              'MAE: {:.3f}, NAE: {:.3f}, MSE: {:.3f}, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), mae.item() / len(data_loader),
                                     nae.item() / len(data_loader), (mse.item() / len(data_loader)) ** 0.5, t_epoch))
        # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.3f}'.format(
        #     epoch, len(data_loader), losses / len(data_loader), t_epoch))

        return losses / len(data_loader), mae.item() / len(data_loader)

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False, epoch=-1, logdir=None):
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
        for batch_idx, (data, gp_gt, M_gt, depthMap, imgs_gt, _) in enumerate(
                data_loader):
            # if map_gt.flatten().max == 0:  # zq
            #     continue
            data, imgs_gt = data, imgs_gt[0]
            B, V, C, H, W = data.shape

            with torch.no_grad():
                imgs_res = self.model(data)
                imgs_res = imgs_res.reshape(V, -1, 1, imgs_gt.shape[-2], imgs_gt.shape[-1])
                imgs_gt = imgs_gt.reshape(V, -1, 1, imgs_gt.shape[-2], imgs_gt.shape[-1]).to(imgs_res.device)

            # #
            # plt.imshow(imgs_res[0, 0].cpu().permute(1, 2, 0))
            # plt.show()

            loss = F.mse_loss(imgs_res[0], imgs_gt.float().to(imgs_res.device))

            target_sum = torch.sum(imgs_gt, dim=(2, 3, 4)) / self.args.density_scale
            res_sum = torch.sum(imgs_res, dim=(2, 3, 4)) / self.args.density_scale

            metric_abs = torch.abs(target_sum - res_sum)
            MAE = torch.mean(metric_abs)
            NAE = torch.mean(metric_abs / torch.where(target_sum == 0, metric_abs, target_sum))
            # RMSE
            MSE = torch.mean((target_sum - res_sum) ** 2)

            # rows, col = 3, 3
            #
            # fig, axes = plt.subplots(nrows=rows, ncols=col, figsize=(16, 8))  # figsize 控制画布大小
            # # 遍历所有子图并绘图
            # for i in range(rows):
            #     ax = axes[i, 0]  # 获取当前子图
            #     ax.imshow(data[0, i].cpu().permute(1, 2, 0))
            #     # data.reshape(V, -1, 3, data.shape[-2], data.shape[-1]).to(imgs_res.device).shape
            #     ax = axes[i, 1]  # 获取当前子图
            #     ax.imshow(imgs_res[i, 0].cpu().permute(1, 2, 0).squeeze())
            #     ax = axes[i, 2]  # 获取当前子图
            #     ax.imshow(imgs_gt[i, 0].cpu().permute(1, 2, 0).squeeze())
            # # 调整子图间距
            # plt.title([f'{x.item():.3f}' for x in metric_abs])
            # plt.tight_layout()
            # plt.show()

            # GAMES_2D_A = (torch.clamp(map_res[:, 0:1].cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2,
            #                                                                                          0).numpy() / 100
            # GAMES_2D_B = (torch.clamp(map_gt[:, 0:1].cpu().detach(), min=0)).squeeze(dim=0).permute(1, 2,
            #                                                                                         0).numpy() / 100
            # for i in range(3):
            #     GAMES_2D[i] += GAME_metric(GAMES_2D_A, GAMES_2D_B, i)

            losses += loss.item()
            mae += MAE
            nae += NAE
            mse += MSE

            # pred = (map_res > self.cls_thres).int().to(map_gt.device)
            # true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            # false_positive = pred.sum().item() - true_positive
            # false_negative = map_gt.sum().item() - true_positive
            # precision = true_positive / (true_positive + false_positive + 1e-4)
            # recall = true_positive / (true_positive + false_negative + 1e-4)
            # precision_s.update(precision)
            # recall_s.update(recall)
            if (batch_idx + 1) % 100 == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Epoch: {}, Batch:{}, Loss: {:.6f}, '
                      'MAE: {:.3f}, NAE: {:.3f}, MSE: {:.3f} Time: {:.1f}, max: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), mae.item() / (batch_idx + 1),
                                        nae.item() / (batch_idx + 1),
                                        (mse.item() / (batch_idx + 1)) ** 0.5, t_epoch, imgs_res.max()))

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(121, title="output")
            subplt1 = fig.add_subplot(122, title="target")
            subplt0.imshow(imgs_res[0:1, 0:1].cpu().detach().numpy().squeeze())
            subplt1.imshow(imgs_gt[0:1, 0:1].cpu().detach().numpy().squeeze())
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)

            # visualizing the heatmap for per-view estimation
            heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
            # heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
            # foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            # foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        if epoch == -1 or epoch == self.args.epochs:

            sample_count = len(data_loader)

            single_view_losses = losses / sample_count

            single_view_mae = mae.item() / sample_count
            single_view_nae = nae.item() / sample_count
            single_view_mse = np.sqrt(mse.item() / sample_count)

            def save_str_to_file(data, root_path, file_name):
                path = os.path.join(root_path, file_name)
                with open(path, 'a+') as f:
                    f.write(data + '\n')

            # if save_result:
            save_reGTAll = 'reGTAll'
            if not os.path.exists(save_reGTAll):
                os.mkdir(save_reGTAll)
            if self.args.name:
                order = self.args.name
            else:
                order = 'tmp'

            save_mae_single_view = 'logdir: {:}, MAE: {:.3f}, NAE: {:.3f}, MSE: {:.3f}, name: {:}, loss: {:}, Time: {:.3f}'.format(
                logdir,
                single_view_mae,
                single_view_nae,
                single_view_mse,
                order,
                single_view_losses,
                t_epoch)

            save_str_to_file(save_mae_single_view, save_reGTAll, f'metrics_single_view.txt')


        len_data_loader = len(data_loader)
        t1 = time.time()
        t_epoch = t1 - t0

        single_view_losses = losses / len_data_loader
        losses = single_view_losses
        single_view_MAE = mae.item() / len_data_loader
        single_view_nae = nae.item() / len_data_loader
        single_view_mse = np.sqrt(mse.item() / len_data_loader)

        self.tb_writer.add_scalar('loss/test', losses, epoch)
        self.tb_writer.add_scalar('single_view_loss/test', losses, epoch)
        self.tb_writer.add_scalar('single_view_mae/test', single_view_MAE, epoch)
        self.tb_writer.add_scalar('single_view_nae/test', single_view_nae, epoch)
        self.tb_writer.add_scalar('single_view_mse/test', single_view_mse, epoch)
        self.tb_writer.add_scalar('time/test', int(t_epoch), epoch)

        print('Test, Epo: {}, Loss: {:.6f}, MAE: {:.3f}, NAE: {:.3f}, MSE: {:.3f}'
              '\tTime: {:.3f}'.format(epoch,
            losses / (len(data_loader) + 1), mae.item() / (len(data_loader) + 1), nae.item() / (len(data_loader) + 1),
            (mse.item() / (len(data_loader) + 1)) ** 0.5, t_epoch))
        # print('Test, Loss: {:.6f}, Time: {:.3f}'.format(losses / (len(data_loader) + 1), t_epoch))

        return losses / len(data_loader), mae.item() / (len(data_loader) + 1), mse.item() / (len(data_loader) + 1)


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
