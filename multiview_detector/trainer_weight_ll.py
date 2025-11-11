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

def print_model_parameters(model):
    for name, parms in model.named_parameters():
        if parms.grad is None or parms.data is None:
            if parms.grad is None and parms.data is not None:
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                      torch.mean(parms.data))
            elif parms.data is None and parms.grad is not None:
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                      ' -->grad_value:', torch.mean(parms.grad))
            else:
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
        else:
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
                  torch.mean(parms.data),
                  ' -->grad_value:', torch.mean(parms.grad))


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer_weight(BaseTrainer):
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
        losses, losses1, losses2 = 0, 0, 0
        mae, nae, mse = 0, 0, 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        loss1_ratio, loss2_ratio = self.args.multi_view_loss1_ratio, self.args.multi_view_loss2_ratio

        sample_len = len(data_loader)

        for batch_idx, (data, map_gt, M_gt, depthMap, imgs_gt, imgs_single_C) in enumerate(
                data_loader):
            # if (batch_idx + 1) % 30 == 0:
            #     continue
            optimizer.zero_grad()
            map_res, M_pre, dist_score = self.model(data, depthMap, train=False, imgs_single_C=imgs_single_C)
            # imgs_res = self.model(data, camera_paras, wld_map_paras, hw_random)
            t_f = time.time()
            t_forward += t_f - t_b
            loss1 = 0
            loss2 = 0

            map_gt = torch.reshape(map_gt, (1, -1)).float()
            gt = torch.sum(map_gt, dim=-1, keepdim=True).to(map_res.device)
            loss1 = F.mse_loss(map_res, gt)

            M_gt = M_gt[0].to(M_pre.device).float()
            loss2 = self.mae_loss(torch.mul(M_pre, M_gt), M_gt)

            # rows, col = 3, 4
            #
            # fig, axes = plt.subplots(nrows=rows, ncols=col, figsize=(16, 8))  # figsize 控制画布大小
            # # 遍历所有子图并绘图
            # for i in range(rows):
            #     for j in range(col):
            #         ax = axes[i, j]  # 获取当前子图
            #         if j == 0:
            #             # continue
            #             ax.imshow(data[0][i].cpu().permute(1, 2, 0))
            #         elif j == 1:
            #             ax.imshow(imgs_single_C[0][i].cpu().permute(1, 2, 0))
            #             pass
            #         elif j == 3:
            #             ax.imshow(M_gt[i, 0].cpu())
            #         else:
            #             ax.imshow(M_gt[i, 1].cpu())
            #         # ax.imshow(data[0, i * col + j].cpu().permute(1, 2, 0))
            #
            # # 调整子图间距
            # plt.tight_layout()
            # # plt.savefig(os.path.join(self.logdir, f'imgs_{epoch}_{batch_idx}.png'))
            # plt.show()

            # plt.imshow(M_gt[0, 0])
            # plt.show()
            # loss = 0.1 * loss1
            loss1 = loss1_ratio * loss1
            loss2 = loss2_ratio * loss2
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()


            target_sum = torch.sum(torch.clamp(map_gt.detach().cpu(), min=0))
            # target_all_sum = torch.sum(torch.clamp(map_gt_all.detach().cpu(), min=0) / 10)
            res_sum = torch.sum(torch.clamp(map_res.detach().cpu(), min=0))
            if target_sum != 0:
                MAE = torch.abs(target_sum - res_sum)
                NAE = torch.abs(target_sum - res_sum) / (target_sum + 1e-18)
                # RMSE
                MSE = (target_sum - res_sum) ** 2
            else:
                MAE = res_sum
                NAE = 1
                # RMSE
                MSE = res_sum ** 2
            losses += loss.item()
            losses1 += loss1.item()
            losses2 += loss2.item()
            mae += MAE
            nae += NAE
            mse += MSE

            t_b = time.time()
            t_backward += t_b - t_f

            if (batch_idx + 1) % 200 == 0 or (len(data_loader) < 200 and (batch_idx + 1) % 20 == 0):
                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="output")
                subplt1 = fig.add_subplot(122, title="target")
                subplt0.imshow(dist_score[0, 0, :, :].cpu().detach().numpy().squeeze())
                subplt1.imshow(dist_score[1, 0, :, :].cpu().detach().numpy())
                plt.savefig(os.path.join(self.logdir, f'train_wei_{str(batch_idx + 1)}.jpg'))
                plt.close(fig)

                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="output")
                subplt1 = fig.add_subplot(122, title="target")
                subplt0.imshow(M_pre[0, 0].cpu().detach().numpy())
                subplt1.imshow(M_gt[0, 0].cpu().detach().numpy())
                plt.savefig(os.path.join(self.logdir, f'train_Mij_{str(batch_idx + 1)}.jpg'))
                plt.close(fig)

            if (batch_idx + 1) % 500 == 0:
                print_model_parameters(self.model)
            #     print("----------------------batch_idx: " + str(batch_idx+1) + " -------------------------------")
            #     print("MAE: " + str(MAE) + " NAE: " + str(NAE))
            #     print("res: " + str(map_res) + " gt: " + str(gt))
            #     for name, parms in self.model.named_parameters():
            #         # if parms.gard > 10000:
            #         #     print(1)
            #         if parms.grad is None or parms.data is None:
            #             pass
            #             # if parms.grad is None and parms.data is not None:
            #             #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #             #           torch.mean(parms.data))
            #             # elif parms.data is None and parms.grad is not None:
            #             #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #             #           ' -->grad_value:', torch.mean(parms.grad))
            #             # else:
            #             #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
            #         else:
            #             print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #                   torch.mean(parms.data),
            #                   ' -->grad_value:', torch.mean(parms.grad))


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
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Loss1: {:.6f}, Loss2: {:.6f}, lr: {:.6f}, '
                      'MAE: {:.3f}, NAE: {:.3f}, RMSE: {:.3f} Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), losses1 / (batch_idx + 1), losses2 / (batch_idx + 1),
                            current_lr, mae.item() / (batch_idx + 1),
                            nae.item() / (batch_idx + 1),
                            (mse.item() / (batch_idx + 1)) ** 0.5, t_epoch, t_forward / batch_idx,
                            t_backward / batch_idx, map_res.max()))
                # print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Time: {:.1f} (f{:.3f}+b{:.3f})'.format(
                #     epoch, (batch_idx + 1), losses / (batch_idx + 1), t_epoch, t_forward / batch_idx, t_backward / batch_idx))
                pass
            # for name, parms in self.model.named_parameters():
            #     if parms.grad is None or parms.data is None:
            #         if parms.grad is None and parms.data is not None:
            #             print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #                   torch.mean(parms.data))
            #         elif parms.data is None and parms.grad is not None:
            #             print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #                   ' -->grad_value:', torch.mean(parms.grad))
            #         else:
            #             print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
            #     else:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight',
            #               torch.mean(parms.data),
            #               ' -->grad_value:', torch.mean(parms.grad))
        if cyclic_scheduler is not None:
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
                cyclic_scheduler.step()

        len_data_loader = len(data_loader)
        t1 = time.time()
        t_epoch = t1 - t0

        single_view_losses = losses / len_data_loader
        losses = single_view_losses
        losses1 = losses1 / len_data_loader
        losses2 = losses2 / len_data_loader
        single_view_MAE = mae.item() / len_data_loader
        single_view_nae = nae.item() / len_data_loader
        single_view_mse = np.sqrt(mse.item() / len_data_loader)

        self.tb_writer.add_scalar('loss/train', losses, epoch)
        self.tb_writer.add_scalar('loss2/train', losses2, epoch)
        self.tb_writer.add_scalar('loss1/train', losses1, epoch)
        self.tb_writer.add_scalar('multi_view_loss/train', losses, epoch)
        self.tb_writer.add_scalar('multi_view_mae/train', single_view_MAE, epoch)
        self.tb_writer.add_scalar('multi_view_nae/train', single_view_nae, epoch)
        self.tb_writer.add_scalar('multi_view_mse/train', single_view_mse, epoch)
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
        for batch_idx, (data, map_gt, M_gt, depthMap, imgs_gt, imgs_single_C) in enumerate(
                data_loader):
            if map_gt.flatten().max == 0:  # zq
                continue

            with torch.no_grad():
                map_res, _, _ = self.model(data, depthMap, train=False, imgs_single_C=imgs_single_C)
                # imgs_res = self.model(data, camera_paras, wld_map_paras, hw_random)
                # print(map_res.shape)
                # print(map_gt.shape)
            # if (batch_idx + 1) % 2000 == 0 or (len(data_loader) < 200 and (batch_idx + 1) % 20 == 0):
            #     fig = plt.figure()
            #     subplt0 = fig.add_subplot(121, title="output")
            #     subplt1 = fig.add_subplot(122, title="target")
            #     subplt0.imshow(map_res[:, 0:1].cpu().detach().numpy().squeeze())
            #     subplt1.imshow(map_gt[:, 0:1].cpu().detach().numpy().squeeze())
            #     plt.savefig(os.path.join(self.logdir, f'test_map_{str(batch_idx + 1)}.jpg'))
            #     plt.close(fig)
            #
            # if (batch_idx + 1) % 20 == 0:
            #     fig = plt.figure()
            #     subplt0 = fig.add_subplot(121, title="output")
            #     subplt1 = fig.add_subplot(122, title="target")
            #     subplt0.imshow(imgs_res[0][0, 0, :, :].cpu().detach().numpy().squeeze())
            #     subplt1.imshow(imgs_gt[0][0, 0, :, :].cpu().detach().numpy())
            #     plt.savefig(os.path.join(self.logdir, f'test_img_{str(batch_idx + 1)}.jpg'))
            #     plt.close(fig)

            map_gt = map_gt.to(map_res.device)
            loss = abs(map_res.sum() - map_gt.sum())

            target_sum = torch.sum(map_gt.detach().cpu())
            res_sum = torch.sum(map_res.detach().cpu())

            MAE = torch.abs(target_sum - res_sum)
            NAE = torch.abs(target_sum - res_sum) / target_sum
            # RMSE
            MSE = (target_sum - res_sum) ** 2

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
                      'MAE: {:.3f}, NAE: {:.3f}, MSE: {:.3f} Time: {:.1f}, maxima: {:.3f}'.format(
                    1, (batch_idx + 1), losses / (batch_idx + 1), mae.item() / (batch_idx + 1),
                                        nae.item() / (batch_idx + 1),
                                        (mse.item() / (batch_idx + 1)) ** 0.5, t_epoch, map_res.max()))

        t1 = time.time()
        t_epoch = t1 - t0

        # if visualize:
        #     fig = plt.figure()
        #     subplt0 = fig.add_subplot(121, title="output")
        #     subplt1 = fig.add_subplot(122, title="target")
        #     subplt0.imshow(imgs_res[0:1, 0:1].cpu().detach().numpy().squeeze())
        #     subplt1.imshow(imgs_gt[0:1, 0:1].cpu().detach().numpy().squeeze())
        #     plt.savefig(os.path.join(self.logdir, 'map.jpg'))
        #     plt.close(fig)
        #
        #     # visualizing the heatmap for per-view estimation
        #     heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
        #     # heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
        #     img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
        #     img0 = Image.fromarray((img0 * 255).astype('uint8'))
        #     head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
        #     head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
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

            save_str_to_file(save_mae_single_view, save_reGTAll, f'metrics_multi_view.txt')


        len_data_loader = len(data_loader)
        t1 = time.time()
        t_epoch = t1 - t0

        single_view_losses = losses / len_data_loader
        losses = single_view_losses
        single_view_MAE = mae.item() / len_data_loader
        single_view_nae = nae.item() / len_data_loader
        single_view_mse = np.sqrt(mse.item() / len_data_loader)

        self.tb_writer.add_scalar('loss/test', losses, epoch)
        self.tb_writer.add_scalar('multi_view_loss/test', losses, epoch)
        self.tb_writer.add_scalar('multi_view_mae/test', single_view_MAE, epoch)
        self.tb_writer.add_scalar('multi_view_nae/test', single_view_nae, epoch)
        self.tb_writer.add_scalar('multi_view_mse/test', single_view_mse, epoch)
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
