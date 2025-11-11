# 导入包
import argparse
import numpy as np
import cv2 as cv
import os
import torch
import json
import sys
import random
from multiview_detector.utils.logger import Logger
from traditional import visualResult

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['OMP_NUM_THREADS'] = '1'

from traditional.EValue import getEValue_ablation as getEValue
from traditional.EValue import getSimilaritySingle, read_json_view, read_json_frame, spatial_transoformation_layer, \
    getSimilarityGroup
from traditional.EValue import getViewDiverse_albation as getViewDiverse
import time


class LbSelect:
    def __init__(self):
        self.scene = None
        self.label_name = None
        self.device = ['cuda:0', 'cuda:0']

    def run(self, args, root_dir):
        print('root_dir: ', root_dir)
        for train in [False]:
            # image
            data_file = '/mnt/d/data/CVCS_dataset/'
            # json
            label_file = '/mnt/d/data/CVCS_dataset/labels/100frames_labels_reproduce_640_480_CVCS/'
            startTime = time.time()
            if train:
                print('training: ')

                file_path = data_file + 'train/'
                label_file_path = label_file + 'train/'
            else:
                file_path = data_file + 'val/'
                label_file_path = label_file + 'val/'

            if train and args.visualNow:
                visualResult.run(root_dir, train, args.single_scene, maxViews=args.maxViews)
                continue

            scene_name_list = os.listdir(file_path)
            nb_scenes = len(scene_name_list)

            l1 = []

            scene_view_init_seed = list(range(nb_scenes))

            for scene_index in range(nb_scenes):
                if not args.single_scene:
                    scene_i = scene_name_list[scene_index]
                else:
                    scene_i = args.scene
                # scene_i = 'scene_02'
                # scene_index = nb_scenes
                # if train and scene_i == 'scene_06':
                #     continue
                self.scene = scene_i
                print('scene: ', scene_i)
                scene_path = os.path.join(file_path, scene_i)
                scene_path_label = os.path.join(label_file_path, scene_i)
                frame_0 = '0'
                label_path0 = os.path.join(label_file_path, scene_i, frame_0, 'json_paras/')
                label_path_list0 = os.listdir(label_path0)
                nb_json = len(label_path_list0)

                frame_path = os.path.join(scene_path, frame_0)
                img_path = frame_path + '/jpgs/'

                allProjectImg = []
                allCameraLoc = []
                allCameraLoc_unnormalize = []
                allCameraRot = []
                allCoverRate = []
                all_wld_map_paras = []
                all_camera_paras = []

                # read all the img and jsonFile in current scene frame 0
                for idx, label_name in enumerate(label_path_list0):
                    self.label_name = label_name
                    img_name = label_name[0:-5] + '.jpg'
                    img_path_name = os.path.join(img_path, img_name)
                    label_path_name = os.path.join(label_path0, label_name)

                    print('\r%d/%d' % (idx + 1, len(label_path_list0)), end='')
                    with open(label_path_name, 'r') as data_file:
                        coords_info = json.load(data_file)

                    # (-elevation, 0, azimuth)
                    fine, _, theta = coords_info['camera']['rot']
                    fine = fine * np.pi / 180
                    theta = theta * np.pi / 180
                    fine = -fine
                    # to unit rotation vector
                    # x: cos_azimuth * cos_elevation
                    # y: sin_azimuth * cos_elevation
                    # z: sin_elevation
                    # number x to angle: x * pi / 180
                    camera_rot = [np.cos(theta) * np.cos(fine),
                                  np.sin(theta) * np.cos(fine),
                                  np.sin(fine)]
                    camera_rot = [float('%.6f' % x) for x in camera_rot]

                    allCameraLoc.append(coords_info['camera']['loc'])
                    allCameraLoc_unnormalize.append(coords_info['camera']['loc'])
                    allCameraRot.append(camera_rot)

                    # r 为比例因子
                    r, a, b, cropped_size = 5, 5, 5, [0, 0]
                    view_size = 1
                    patch_num = 1
                    batch_size = 1

                    coords_3d_id_all, wld_map_paras = read_json_frame(coords_info, r, a, b)
                    coords_3d_id_all, coords_2d_all, camera_paras = read_json_view(coords_info)

                    r, a, b, cropped_size, patch_num = wld_map_paras[0], wld_map_paras[1], wld_map_paras[
                        2], wld_map_paras[3:5], 1

                    paras = [batch_size, view_size, patch_num, cropped_size]
                    camera_paras2_shape = (batch_size * view_size, 15)
                    camera_paras2 = torch.reshape(torch.Tensor(camera_paras), shape=camera_paras2_shape)

                    img_feature = cv.imread(img_path_name)
                    img_feature = cv.resize(img_feature, (img_feature.shape[0] // 3, img_feature.shape[1] // 3))
                    img_feature = torch.Tensor(img_feature).reshape(
                        ([-1, img_feature.shape[0], img_feature.shape[1], img_feature.shape[2]]))
                    img_feature = img_feature.clone().permute(dims=(0, 3, 1, 2))

                    wld_map_paras = torch.Tensor(wld_map_paras).reshape((1, -1))

                    res = spatial_transoformation_layer(scene_i, paras,  # [1, 5, 5, (h_actual, w_actual)]
                                                        [img_feature.to(self.device[0]),  # 'cuda:0'
                                                         camera_paras2.to(self.device[0]),  # 'cuda:0'
                                                         wld_map_paras.to(self.device[0])  # 'cuda:0'
                                                         ])

                    resImg = torch.norm(res[0].detach(), dim=0).cpu().numpy()
                    allProjectImg.append(resImg)
                    all_camera_paras.append(camera_paras2)
                    all_wld_map_paras.append(wld_map_paras)
                    allCoverRate.append(wld_map_paras[0, 3] * wld_map_paras[0, 4])
                # break

                # max min normalize the camera location
                allCameraLoc = np.array(allCameraLoc)
                allCameraLoc_unnormalize = np.array(allCameraLoc_unnormalize)
                allCameraLoc = (allCameraLoc - allCameraLoc.min(axis=0)) / (
                        allCameraLoc.max(axis=0) - allCameraLoc.min(axis=0))

                all_camera_paras = torch.Tensor(np.array(all_camera_paras)).squeeze()
                # all_camera_paras = torch.Tensor(all_camera_paras)
                all_wld_map_paras = torch.Tensor(np.array(all_wld_map_paras)).squeeze()
                # all_wld_map_paras = torch.Tensor(all_wld_map_paras)
                allCameraLoc = torch.from_numpy(allCameraLoc)
                allCameraRot = torch.Tensor(allCameraRot)

                # 各个视角在整个地平面下的掩码
                allProjectImgBi = np.zeros((1, allProjectImg[0].shape[0], allProjectImg[0].shape[1]))
                for img in allProjectImg:
                    img = np.where(img < 1, 0, 1)
                    img = np.expand_dims(img, axis=0)
                    allProjectImgBi = np.concatenate((allProjectImgBi, img), axis=0)
                allProjectImgBi = torch.Tensor(allProjectImgBi[1:])
                allProjectImgArea = torch.sum(allProjectImgBi.cuda(), dim=[1, 2]).cpu()
                allCoverRate = allProjectImgArea / torch.tensor(allCoverRate)
                # 预处理完成，获取所有图片投影到地平面的掩码，以及其面积

                # 排序
                #  按allProjectImgArea的值进行排序，但返回的是在allProjectImgArea中的索引

                if args.randomInit:
                    torch.manual_seed(scene_view_init_seed[scene_index])
                    accessAreaIds = list(torch.randperm(nb_json).numpy())
                else:
                    accessAreaIds = list(np.array(torch.argsort(allProjectImgArea, descending=True)))  # 其面积按照索引顺序依次递减

                # step1：初始化视角，maxE1
                maxViews = args.maxViews
                maxE1 = -1
                currentIdx = maxViews  # 指向下一个要访问的视角索引
                currentGroup = accessAreaIds[:maxViews]
                maxE1Group = currentGroup
                maxSimilarity = -1
                allViewsNum = len(allProjectImgArea)

                while True:
                    e1 = getEValue(allProjectImgBi, currentGroup, all_wld_map_paras, allCameraLoc, allCameraRot,
                                   args, allCameraLoc_unnormalize)
                    if e1 > maxE1:
                        maxE1 = e1
                        maxE1Group = currentGroup.copy()
                        maxSimilarity = getSimilarityGroup(currentGroup, allCameraLoc, allCameraRot, args)
                    else:
                        break

                    # step3: delete view
                    minViewDiverse = np.inf
                    minViewDiverseIdx = 0
                    for idx in range(maxViews):
                        viewDiverse = getViewDiverse(idx, allProjectImgBi, currentGroup, all_wld_map_paras,
                                                     allCameraLoc, allCameraRot, args, allCameraLoc_unnormalize,
                                                     allCoverRate,
                                                     label_path_list0)
                        if viewDiverse < minViewDiverse:
                            minViewDiverse = viewDiverse
                            minViewDiverseIdx = idx
                    print('deleteView: ', label_path_list0[currentGroup[minViewDiverseIdx]])
                    tmpDelView = currentGroup.pop(minViewDiverseIdx)

                    tmpMaxE1 = -1
                    tmpMaxE1Idx = 0
                    for tmpViewIdx in accessAreaIds:
                        if tmpViewIdx in currentGroup or tmpViewIdx == tmpDelView:
                            continue
                        currentGroup.append(tmpViewIdx)
                        print('currentGroup: ', [label_path_list0[x] for x in currentGroup])
                        tmpE1 = getEValue(allProjectImgBi, currentGroup, all_wld_map_paras, allCameraLoc,
                                          allCameraRot,
                                          args, allCameraLoc_unnormalize)
                        if tmpMaxE1 < tmpE1:
                            tmpMaxE1 = tmpE1
                            tmpMaxE1Idx = tmpViewIdx
                        currentGroup.pop()
                    currentGroup.append(tmpMaxE1Idx)
                    print('currentGroup after replace: ', [label_path_list0[x] for x in currentGroup])

                    currentIdx += 1
                    print('-' * 20)
                print('result: ', [label_path_list0[x] for x in maxE1Group])
                print('*' * 100)
                # scene, frame, maxViews, 5 * maxViews, similarity
                save_tag = []
                save_tag.append(int(scene_i[6:]))
                save_tag.append(int(frame_0))
                save_tag += [int(label_path_list0[x][:-5]) for x in maxE1Group]
                for _ in range(5):
                    save_tag += [int(x[:-5]) for x in random.sample(label_path_list0, k=args.maxViews)]
                # save_tag.append()
                save_tag.append(maxSimilarity)
                l1.append(list(save_tag))
                # torch.cuda.empty_cache()
                if args.single_scene:
                    break
            l1 = np.array(l1)
            if train:
                np.savetxt(os.path.join(root_dir, 'result_train.txt'), l1, fmt="%.6f")
            else:
                np.savetxt(os.path.join(root_dir, 'result_val.txt'), l1, fmt="%d")
            endTime = time.time()
            print('calculate spend time(s): ', str('%.2f' % (endTime - startTime)))
            startTime = time.time()
            visualResult.run(root_dir, train, args.single_scene, maxViews=maxViews)
            endTime = time.time()
            print('visualize spend time(s): ', str('%.2f' % (endTime - startTime)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--eM', type=str, default='e4', help='ratio for per view loss')
    parser.add_argument('--addView', type=str, default='e3')
    parser.add_argument('--delViews', type=str, default='e3')
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--maxViews', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--distWay', type=str, default='e1')
    parser.add_argument('--save_dir', type=str, default='exp3/maxView_ablation/',
                        help='ratio for per view loss')
    parser.add_argument('--scene', type=str, default='scene_03')
    parser.add_argument('--single_scene', type=bool, default=False)
    parser.add_argument('--randomInit', type=bool, default=False)
    parser.add_argument('--visualNow', type=bool, default=False)

    delView = ['e1', 'e2', 'e3']
    titleDelV = ['delByS', 'delBySAndCoverRate', 'delByViewEstimation']
    # delView = ['e1', 'e2']
    # eM = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
    eM = ['e1', 'e2', 'e5', 'e6']
    titleEM = ['similarity', 'similarity_distance', 'similarity_distance_coverrate',
               'similarity_meandistance_coverrate']

    # titleEM = ['similarity', 'similarity_distance', 'similarity_meandistance', 'similarity_coverrate',
    #            'similarity_distance_coverrate', 'similarity_meandistance_coverrate']
    # maxViews = [4, 3, 2]
    maxViews = [5]

    randomInit = [False]
    distWay = ['e1', 'e2']
    distTitle = ['reciDist', 'expDist']

    for views in maxViews:
        for i1 in randomInit:
            for i2 in range(len(eM)):
                for i3 in range(len(delView)):
                    if i2 < 2:
                        parser.set_defaults(maxViews=views)
                        parser.set_defaults(delViews=delView[i3])
                        parser.set_defaults(eM=eM[i2])
                        parser.set_defaults(randomInit=i1)
                        args = parser.parse_args()
                        root_dir = args.save_dir + "/maxView" + str(views) + '/' + titleEM[i2] + '_' + titleDelV[i3]
                        sys.stdout = Logger(os.path.join(root_dir, 'log_train_val2.txt'), )

                        me = LbSelect()
                        me.run(args, root_dir)
                    else:
                        for i4 in range(len(distWay)):
                            parser.set_defaults(maxViews=views)
                            parser.set_defaults(delViews=delView[i3])
                            parser.set_defaults(eM=eM[i2])
                            parser.set_defaults(randomInit=i1)
                            parser.set_defaults(distWay=distWay[i4])
                            args = parser.parse_args()
                            # root_dir = args.save_dir + '_randomInit' + str(i1) + '_' + titleEM[i2] + '_' + titleDelV[i3]
                            root_dir = args.save_dir + "/maxView" + str(views) + '/' + titleEM[i2] + '_' + distTitle[i4] + '_' + titleDelV[i3]
                            sys.stdout = Logger(os.path.join(root_dir, 'log_train_val2.txt'), )

                            me = LbSelect()
                            me.run(args, root_dir)

    # parser = argparse.ArgumentParser(description='Multiview detector')
    # parser.add_argument('--eM', type=str, default='e4', help='ratio for per view loss')
    # parser.add_argument('--addView', type=str, default='e3')
    # parser.add_argument('--delViews', type=str, default='e3')
    # parser.add_argument('--lambda1', type=float, default=0.1)
    # parser.add_argument('--maxViews', type=int, default=3)
    # parser.add_argument('--gamma', type=float, default=1.)
    # parser.add_argument('--distWay', type=str, default='e1')
    # parser.add_argument('--save_dir', type=str, default='exp3/maxView2_singleview/',
    #                     help='ratio for per view loss')
    # parser.add_argument('--scene', type=str, default='scene_03')
    # parser.add_argument('--single_scene', type=bool, default=True)
    # parser.add_argument('--randomInit', type=bool, default=False)
    # parser.add_argument('--visualNow', type=bool, default=False)
    #
    # args = parser.parse_args()
    # if args.single_scene:
    #     root_dir = args.save_dir + '_' + args.scene
    # else:
    #     root_dir = args.save_dir
    # sys.stdout = Logger(os.path.join(root_dir, 'log_train_val2.txt'), )
    # me = LbSelect()
    # me.run(args, root_dir)
