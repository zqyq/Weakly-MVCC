import json
import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def file_write_list(l1, path):
    '''
    l1: list[[list],...]
    '''
    # with open(path)
    with open(path, 'w') as f:
        for eleList in l1:
            f.write(' '.join([str(x) for x in eleList]))
            f.write('\n')


def file_read_list(path, dtype=str) -> list:
    """

    the dtype must be basic python data type

    """
    with open(path, 'r') as f:
        l2 = []
        s = f.readline()
        while s is not None and len(s.strip()) != 0:
            s = s.strip()
            if dtype == str:
                l2.append(s.split(' '))
            else:
                l2.append([dtype(x) for x in s.split(' ')])
            s = f.readline()
    return l2

def read_json(coords_info):

    # get people 2D and 3D coords:
    coords = coords_info['image_info']

    coords_2d_all = [] #np.zeros((1, 2))

    # id = 0
    for point in coords:
        id = point['idx']

        coords_2d0 = point['pixel']
        if coords_2d0 != None:
            coords_2d = [coords_2d0[1]/1920.0, coords_2d0[0]/1080.0] + [id]

            coords_2d_all.append(coords_2d)

    # form the para list:
    return coords_2d_all

def _meshgrid(height, width):
    x_linspace = torch.linspace(-1., 1., width)
    y_linspace = torch.linspace(-1., 1., height)

    # x_coordinates, y_coordinates = torch.meshgrid(x_linspace, y_linspace)
    y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)

    x_coordinates = torch.reshape(x_coordinates, [-1])
    y_coordinates = torch.reshape(y_coordinates, [-1])
    ones = torch.ones_like(x_coordinates)

    indices_grid = torch.cat([x_coordinates, y_coordinates, ones], dim=0)
    return indices_grid

def _repeat(x, num_repeats):
    ones = torch.ones((1, num_repeats))
    x = torch.reshape(x, shape=(-1, 1))
    x = torch.matmul(x, ones)
    return torch.reshape(x, [-1])

def _interpolate(image, x, y, output_size):
    # batch_size = tf.shape(image)[0]
    # height = tf.shape(image)[1]
    # width = tf.shape(image)[2]
    # num_channels = tf.shape(image)[3]

    batch_size = image.shape[0]
    height = image.shape[1]
    width = image.shape[2]
    num_channels = image.shape[3]

    # x = torch.tensor(x)
    # y = torch.tensor(y)

    # height_float = torch.cast(height, dtype='float32')
    # width_float = torch.cast(width, dtype='float32')

    height_float = torch.tensor(height)
    width_float = torch.tensor(width)

    output_height = output_size[0]
    output_width = output_size[1]

    x = .5 * (x + 1.0) * (width_float)
    y = .5 * (y + 1.0) * (height_float)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    max_y = height - 1
    max_x = width - 1

    x0 = torch.clamp(x0, 0, max_x).to(image.device)
    x1 = torch.clamp(x1, 0, max_x).to(image.device)
    y0 = torch.clamp(y0, 0, max_y).to(image.device)
    y1 = torch.clamp(y1, 0, max_y).to(image.device)

    flat_image_dimensions = width * height
    pixels_batch = torch.range(0, batch_size-1) * flat_image_dimensions
    flat_output_dimensions = output_height * output_width
    base = _repeat(pixels_batch, flat_output_dimensions).to(image.device)
    base_y0 = base + y0 * width
    base_y1 = base + y1 * width
    indices_a = base_y0 + x0
    indices_b = base_y1 + x0
    indices_c = base_y0 + x1
    indices_d = base_y1 + x1

    flat_image = torch.reshape(image, shape=(-1, num_channels))
    # flat_image = torch.tensor(flat_image)
    pixel_values_a = torch.index_select(flat_image, dim=0, index=indices_a.long())
    pixel_values_b = torch.index_select(flat_image, dim=0, index=indices_b.long())
    pixel_values_c = torch.index_select(flat_image, dim=0, index=indices_c.long())
    pixel_values_d = torch.index_select(flat_image, dim=0, index=indices_d.long())

    # x0 = torch.cast(x0, 'float32')
    # x1 = torch.cast(x1, 'float32')
    # y0 = torch.cast(y0, 'float32')
    # y1 = torch.cast(y1, 'float32')

    area_a = torch.unsqueeze(((x1 - x) * (y1 - y)), 1)
    area_b = torch.unsqueeze(((x1 - x) * (y - y0)), 1)
    area_c = torch.unsqueeze(((x - x0) * (y1 - y)), 1)
    area_d = torch.unsqueeze(((x - x0) * (y - y0)), 1)

    output = area_a * pixel_values_a+area_b * pixel_values_b+area_c * pixel_values_c+area_d * pixel_values_d
    return output

def _transform(feature, affine_transformation, visualized=False):
    b, h, w, c = feature.shape
    # feature = torch.reshape(feature, (b, h, w, c))

    indices_grid = _meshgrid(h, w)
    indices_grid = torch.unsqueeze(indices_grid, dim=0)
    indices_grid = torch.reshape(indices_grid, [-1])  # flatten?

    # indices_grid = torch.tile(indices_grid, (b, 1))
    indices_grid = torch.reshape(indices_grid, (1, 3, -1)).to(feature.device)

    transformed_grid = torch.matmul(affine_transformation.float(), indices_grid.float())
    transformed_grid = torch.divide(transformed_grid[:, :2, :], (transformed_grid[:, 2:, :] + 1e-8))
    # transformed_grid_w = (transformed_grid[:, 0:1, :] * 2.0) / 1920 - 1.0
    # transformed_grid_h = (transformed_grid[:, 1:2, :] * 2.0) / 1080 - 1.0
    #
    # transformed_grid_w = torch.clamp(transformed_grid_w, -10, 10)
    # transformed_grid_h = torch.clamp(transformed_grid_h, -10, 10)
    # transformed_grid_wh = torch.cat([transformed_grid_w, transformed_grid_h], dim=1)

    x_s = transformed_grid[:, 0, :]
    y_s = transformed_grid[:, 1, :]
    # x_s = transformed_grid_wh[:, 0, :]
    # y_s = transformed_grid_wh[:, 1, :]
    x_s_flatten = torch.reshape(x_s, [-1])
    y_s_flatten = torch.reshape(y_s, [-1])

    transformed_image = _interpolate(feature, x_s_flatten, y_s_flatten, [h, w])
    transformed_image = torch.reshape(transformed_image, shape=[-1, h, w, c])

    if visualized:
        plt.imshow(transformed_image[0].int())
        plt.show()
        # plt.imshow(transformed_image.permute(0, 3, 1, 2)[0, 0])
        # plt.show()

    return transformed_image.permute(0, 3, 1, 2)


if __name__ == '__main__':
    img_path1 = r'/mnt/data/Datasets/CVCS/train/scene_80/50/jpgs/111.jpg'
    img_path2 = r'/mnt/data/Datasets/CVCS/train/scene_80/50/jpgs/81.jpg'

    file_path1 = r'/mnt/data/Datasets/CVCS/labels/100frames_labels_reproduce_640_480_CVCS/100frames_labels_reproduce_640_480_CVCS/train/scene_80/50/json_paras/111.json'
    file_path2 = r'/mnt/data/Datasets/CVCS/labels/100frames_labels_reproduce_640_480_CVCS/100frames_labels_reproduce_640_480_CVCS/train/scene_80/50/json_paras/81.json'

    with open(file_path1, 'r') as data_file1:
        coords_info_frame1 = json.load(data_file1)

    with open(file_path2, 'r') as data_file2:
        coords_info_frame2 = json.load(data_file2)

    coords_info1 = coords_info_frame1['image_info']
    coords_info2 = coords_info_frame2['image_info']

    # # distCoeffs1 = np.array(coords_info_frame1['distCoeffs'], dtype=float)
    # # cameraMatrix1 = np.asarray(coords_info_frame1['cameraMatrix'])
    # #
    # # distCoeffs2 = np.array(coords_info_frame2['distCoeffs'], dtype=float)
    # # cameraMatrix2 = np.asarray(coords_info_frame2['cameraMatrix'])
    #
    src_coords, dst_coords = [], []
    for point1, point2 in zip(coords_info1, coords_info2):
        p1 = point1['pixel']
        p1_idx = point1['idx']
        p2 = point2['pixel']
        p2_idx = point2['idx']

        # if p1 is not None and p2 is not None:
        #     src_coords.append((p1[1], p1[0]))
        #     dst_coords.append((p2[1], p2[0]))
            # src_coords.append((p1[0] / 1080, p1[1] / 1920))
            # dst_coords.append((p2[0] / 1080, p2[1] / 1920))
        if p1 is not None:
            src_coords.append([p1[1] / 1920, p1[0] / 1080] + [p1_idx])
        if p2 is not None:
            dst_coords.append([p2[1] / 1920, p2[0] / 1080] + [p2_idx])


    src_coords = np.asarray(src_coords, dtype=float)
    # # src_coords = np.expand_dims(src_coords, axis=1)
    # # un_src_coords = cv2.undistortPoints(src_coords, cameraMatrix1, distCoeffs1, P=cameraMatrix1)
    # # un_src_coords = np.reshape(un_src_coords, (un_src_coords.shape[0], un_src_coords.shape[2]))
    # # un_src_coords[:, 0] *= 1080
    # # un_src_coords[:, 1] *= 1920
    #
    dst_coords = np.array(dst_coords, dtype=float)
    # # dst_coords = np.expand_dims(dst_coords, axis=1)
    # # un_dst_coords = cv2.undistortPoints(dst_coords, cameraMatrix2, distCoeffs2, P=cameraMatrix2)
    # # un_dst_coords = np.reshape(un_dst_coords, (un_dst_coords.shape[0], un_dst_coords.shape[2]))
    # # un_dst_coords[:, 0] *= 1080
    # # un_dst_coords[:, 1] *= 1920
    id_list1 = list(src_coords[:, 2])
    id_list2 = list(dst_coords[:, 2])
    id_list12 = list(set(id_list1).intersection(id_list2))

    v1_pmap_i_12 = [src_coords[src_coords[:, 2] == id_list12[i], :] for i in range(len(id_list12))]
    v2_pmap_i_12 = [dst_coords[dst_coords[:, 2] == id_list12[i], :] for i in range(len(id_list12))]
    v1_pmap_i_12 = np.concatenate(v1_pmap_i_12, axis=0)[:, :2]
    v2_pmap_i_12 = np.concatenate(v2_pmap_i_12, axis=0)[:, :2]
    src_coords, dst_coords = v1_pmap_i_12, v2_pmap_i_12

    v1_pmap_i_12[:, 0] = (v1_pmap_i_12[:, 0] - 0.5) / 0.5
    v1_pmap_i_12[:, 1] = (v1_pmap_i_12[:, 1] - 0.5) / 0.5

    v2_pmap_i_12[:, 0] = (v2_pmap_i_12[:, 0] - 0.5) / 0.5
    v2_pmap_i_12[:, 1] = (v2_pmap_i_12[:, 1] - 0.5) / 0.5

    src_pts = np.float32(v1_pmap_i_12).reshape(-1, 1, 2)
    dst_pts = np.float32(v2_pmap_i_12).reshape(-1, 1, 2)

    # matrix_s2d, mask = cv2.findHomography(src_coords, dst_coords, 0)
    # # matrix_d2s, mask = cv2.findHomography(dst_coords, src_coords, 0)
    # # matrix, mask = cv2.findHomography(un_src_coords, un_dst_coords, 0)
    matrix_s2d, mask = cv2.findHomography(src_pts, dst_pts, 0)

    # v1_label = file_path1
    # v2_label = file_path2
    # v1_pmap_12, v2_pmap_12 = [], []
    #
    # # read labels:
    # with open(v1_label, 'r') as data_file:
    #     coords_info1 = json.load(data_file)
    # coords_2d_v1 = read_json(coords_info1)
    # with open(v2_label, 'r') as data_file:
    #     coords_info2 = json.load(data_file)
    # coords_2d_v2 = read_json(coords_info2)
    #
    # v1_pmap_i = np.asarray(coords_2d_v1)
    # v2_pmap_i = np.asarray(coords_2d_v2)
    # if coords_2d_v1 != [] and coords_2d_v2 != []:
    #     id_list1 = list(v1_pmap_i[:, 2])
    #     id_list2 = list(v2_pmap_i[:, 2])
    #     id_list12_f = list(set(id_list1).intersection(id_list2))
    #
    #     v1_pmap_i_12 = [v1_pmap_i[v1_pmap_i[:, 2] == id_list12_f[i], :] for i in
    #                     range(len(id_list12_f))]
    #     v2_pmap_i_12 = [v2_pmap_i[v2_pmap_i[:, 2] == id_list12_f[i], :] for i in
    #                     range(len(id_list12_f))]
    #
    #     v1_pmap_12 = v1_pmap_12 + v1_pmap_i_12
    #     v2_pmap_12 = v2_pmap_12 + v2_pmap_i_12
    #     v1_pmap_12 = np.asarray(v1_pmap_12)
    #     v2_pmap_12 = np.asarray(v2_pmap_12)
    #
    #     if v1_pmap_12 == np.asarray([]) or v2_pmap_12 == np.asarray([]):
    #         matrix_s2d = -np.asarray([-1, -1, -1, -1, -1, -1, 0, 0, 1])
    #     else:
    #         matrix_s2d = cal_homography_frame(v1_pmap_12[:, 0, :2],
    #                                       v2_pmap_12[:, 0, :2])
    img_src = cv2.imread(img_path1)
    img_src = cv2.resize(img_src, (640, 360))
    img_src = torch.asarray(img_src).unsqueeze(0)
    # img_src = torch.ones((1, 360, 640, 1))
    matrix_s2d = torch.from_numpy(matrix_s2d).unsqueeze(0)
    perspective_img1 = _transform(img_src, matrix_s2d, True)
    # perspective_img1 = cv2.warpPerspective(img_src, matrix_s2d, (img_src.shape[1], img_src.shape[0]))
    # plt.imshow(perspective_img1)
    # plt.show()
    #
    # src_coords, dst_coords = np.asarray(src_coords[:, :2]), np.asarray(dst_coords[:, :2])
    # z_coords = np.ones([src_coords.shape[0], 1])
    # un_src_coords = np.concatenate((src_coords, z_coords), axis=1)
    # transfer_coords = np.matmul(matrix_s2d, un_src_coords.T)
    # transfer_coords = transfer_coords.T
    # #
    # dist = []
    # mx_dist = 0
    # for p1, p2 in zip(dst_coords, transfer_coords):
    #     tmp = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    #     mx_dist = max(mx_dist, tmp)
    #     dist.append(tmp)
    #
    # print(dist)
    # print(mx_dist)
    # pass

    # perspective_img = cv2.warpPerspective(img_src, matrix, (img_src.shape[1], img_src.shape[0]))
    # plt.imshow(perspective_img)
    # plt.show()
    # img_dst = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    # perspective_img2 = cv2.warpPerspective(img_dst, matrix_d2s, (img_dst.shape[1], img_dst.shape[0]))
    # plt.imshow(perspective_img2)
    # plt.show()






