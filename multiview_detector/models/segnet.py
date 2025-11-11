# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/7 下午2:02
@Auth ： Fei Xue
@File ： segnet.py
@Email： xuefei@sensetime.com
"""
import os
import os.path as osp
from mmseg.apis import inference_model, init_model
import cv2


# build the model from a config file and a checkpoint file

class SegNet:
    def __init__(self, model_name="deeplabv3plus", device='cuda:0'):
        # abs_path = "/home/Code/Deeplearning/r2d2/nets/semseg"
        seg_path = os.getcwd()
        if model_name == "deeplabv3":
            config_file = 'nets/semseg/configs/deeplabv3/deeplabv3_r50-d8_512x512_80k_ade20k.py'
            checkpoint_file = 'nets/semseg/checkpoints/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'
        elif model_name == "deeplabv3plus":
            config_file = 'nets/semseg/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
            # checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth'
            checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'

        elif model_name == 'convxts-base-ade20k':
            config_file = '../configs/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k.py'
            checkpoint_file = "/mnt/d/DJ/CF/ pretrain_model/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth"

        self.model = init_model(os.path.join(seg_path, config_file), checkpoint_file, device=device)

    def evaluate(self, img):
        result = inference_model(model=self.model, img=img)
        return result

if __name__ == '__main__':
    seg = SegNet(model_name='convxts-base-ade20k', device="cuda:1")
    img_path = "/mnt/d/data/CVCS_dataset/train/scene_04/11/jpgs/29.jpg"
    img = cv2.imread(img_path)
    # tmp = cv2.resize(img, (640, 360))

    img = img[:, :, (2, 1, 0)]  # BGR -> RGB
    img = img.astype('float32')

    # normal
    img = img / 255.0
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
    # downsample
    img = cv2.resize(img, (512, 512))

    seg_result = seg.evaluate(img)
    pass