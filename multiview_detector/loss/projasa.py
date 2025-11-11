import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_sum(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h / h.sum()


def map_counting(heatmap, center, sigma, k=1):
    radius = int(3 * sigma)
    diameter = 2 * radius + 1
    gaussian = gaussian_sum((diameter, diameter), sigma=sigma)

    x, y = int(center[0]), int(center[1])

    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        heatmap[y - top:y + bottom, x - left:x + right] = np.add(masked_heatmap, masked_gaussian)

    return heatmap

if __name__ == '__main__':
    import torch.nn.functional as F

    frame_range = range(636, 1236, 2)
    world_gridshape = (768, 640)
    world_reduce = 2
    hgwg = list(map(lambda x: x // world_reduce, world_gridshape))
    ground_plane_gt = {}
    with h5py.File(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
        grounddata = f['v_pmap_GP']
        for frame in frame_range:
            occupancy_info = (grounddata[grounddata[:, 0] == frame, 2:4])
            occupancy_map = np.zeros(hgwg)
            for idx in range(occupancy_info.shape[0]):
                cx, cy = occupancy_info[idx]
                cx = int(cx // world_reduce)
                cy = int(cy // world_reduce)
                center = (cx, cy)
                draw_umich_gaussian(occupancy_map, center, sigma=5)
            plt.imshow(occupancy_map)
            plt.show()
                # occupancy_map[cy, cx] = 1
            # ground_plane_gt[frame] = occupancy_map
            # target = torch.from_numpy(occupancy_map)[None, None]
            # kernel = prod_kernel(20, 5)
            # map_gt = F.conv2d(target, kernel.double().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
            # plt.imshow(map_gt.squeeze().cpu())
            # plt.show()
            break
