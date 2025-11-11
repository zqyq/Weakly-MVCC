import cv2
import numpy as np
import random


def data_augmentation(image, flip_prob=0.5, rotate_range=30, clip_ratio=0.1, scale_range=(0.7, 1),
                      fill_mode='constant', fill_value=0, visualize=False, seed=None, is_density_map=False):
    """
    数据增强方法，包含翻转、旋转、裁剪、缩放，并保持图像大小不变。

    参数:
        image: 输入图像 (H, W, C)
        flip_prob: 水平翻转概率 (默认0.5)
        rotate_range: 旋转角度范围 (默认±30度)
        clip_ratio: 随机裁剪比例 (默认0.1，即裁剪10%边缘)
        scale_range: 缩放比例范围 (默认0.8~1.2)
        fill_mode: 填充模式 ('constant', 'edge', 'reflect')
        fill_value: 填充常数值 (fill_mode='constant'时生效)

    返回:
        增强后的图像 (H, W, C)
    """
    if seed is None:
        seed = random.randint(1, 10000)

    random.seed(seed)
    np.random.seed(seed)

    h, w = image.shape[:2]
    augmented = image.copy()

    # 1. 随机水平翻转
    if random.random() < flip_prob:
        augmented = cv2.flip(augmented, 1)  # 1表示水平翻转

    # 2. 随机旋转 (±rotate_range度)
    # angle = random.uniform(-rotate_range, rotate_range)
    # M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    # augmented = cv2.warpAffine(augmented, M, (w, h), flags=cv2.INTER_LINEAR,
    #                            borderMode=cv2.BORDER_CONSTANT if fill_mode == 'constant' else cv2.BORDER_REFLECT,
    #                            borderValue=fill_value)


    # 3. 随机裁剪 (保留中心区域)
    clip_pixels = int(min(h, w) * clip_ratio)
    if clip_pixels > 0:
        y_start = random.randint(0, clip_pixels)
        x_start = random.randint(0, clip_pixels)

        cropped = augmented[y_start:h - clip_pixels + y_start, x_start:w - clip_pixels + x_start]

        augmented = np.zeros_like(augmented)
        augmented[y_start:h - clip_pixels + y_start, x_start:w - clip_pixels + x_start] = cropped

    if is_density_map:
        return augmented, seed

    # 4. 随机缩放 (保持图像中心，填充边缘)
    scale = random.uniform(*scale_range)
    if scale != 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(augmented, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充至原尺寸
        pad_h = max(0, h - new_h)
        pad_w = max(0, w - new_w)
        top = pad_h // 2
        left = pad_w // 2

        if fill_mode == 'constant':
            augmented = cv2.copyMakeBorder(scaled, top, pad_h - top, left, pad_w - left,
                                           cv2.BORDER_CONSTANT, value=fill_value)
        elif fill_mode == 'edge':
            augmented = cv2.copyMakeBorder(scaled, top, pad_h - top, left, pad_w - left,
                                           cv2.BORDER_REPLICATE)
        else:  # reflect
            augmented = cv2.copyMakeBorder(scaled, top, pad_h - top, left, pad_w - left,
                                           cv2.BORDER_REFLECT)

    if augmented.shape[0] != h or augmented.shape[1] != w:
        augmented = cv2.resize(augmented, (w, h), interpolation=cv2.INTER_LINEAR)

    if visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        subplt0 = fig.add_subplot(121, title="origin")
        subplt1 = fig.add_subplot(122, title="affined")
        subplt0.imshow(img)
        subplt1.imshow(augmented)
        plt.show()

    return augmented, seed


# 示例使用
if __name__ == "__main__":
    img = cv2.imread("/mnt/d/data/PETS2009/image_frames/Train/S1L3/Time_14-17/View_001/frame_0000.jpg")  # 替换为你的图像路径
    augmented_img,seed = data_augmentation(img)
    pass

    # cv2.imshow("Original", img)
    # cv2.imshow("Augmented", augmented_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()