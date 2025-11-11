#  Bin Li, Daijie Chen, and Qi Zhang.
# WSCF-MVCC: Weakly-supervised  Calibration-free Multi-view Crowd Counting. PRCV 2025

![Pipeline](assets/pipeline.png "Pipeline")

## Abstract
Multi-view crowd counting can effectively mitigate occlusion issues that commonly arise in single-image crowd counting. Existing deep-learning multi-view crowd counting methods project different camera view images onto a common space to obtain ground-plane density maps, requiring abundant and costly crowd annotations and camera calibrations. Hence, calibration-free methods are proposed that do not require camera calibrations and scene-level crowd annotations. However, existing calibration-free methods still require expensive image-level crowd annotations for training the single-view counting module. Thus, in this paper, we propose a weakly-supervised calibration-free multi-view crowd counting method (WSCF-MVCC), directly using crowd count as supervision for the single-view counting module rather than density maps constructed from crowd annotations. Instead, a self-supervised ranking loss that leverages multi-scale priors is utilized to enhance the model’s perceptual ability without additional annotation costs. What’s more, the proposed model leverages semantic information to achieve a more accurate view matching and, consequently, a more precise scene-level crowd count estimation. The proposed method outperforms the state-of-the-art methods on three widely used multi-view counting datasets under weakly supervised settings, indicating that it is more suitable for practical deployment compared with calibrated methods.


## Poster 
AAAI 2024 poster:
![Poster](assets/poster.png "Poster")

## Overview
We release the PyTorch code for the WSCF-MVCC, a weakly-supervised calibration-free multi-view crowd counting method (WSCF-MVCC). 
 
## Content
- [Dependencies](#dependencies)
- [Data Preparation](#Data Preparation)
- [Training](#Training)
- [Perspective transformation](#Perspective transformation)


## Dependencies
- python
- pytorch & torchvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- tqdm
- h5py
- argparse

## Data Preparation
In the code implementation, the root path of the four main datasets is defined as ```/mnt/d/data```. Of course,
it can be changed.
When you apply the method to your datasets or other paths, the root path should look like this:
```
/mnt/d/data/
|__CVCS
    |__...
|__CityStreet
    |__...
|__PETS2009
    |__...
```
## Training
 During the training phase, we need to train the model in 3 stages, the feature extractor is shared across all camera views, i.e., ResNet18 and VGG16. 

Take training a detector on CVCS dataset as an example, to train the final detector, run the following script in order.
```shell script
python main.py -d cvcs --variant 2D 
```
After obtaining the trained 2D feature extractor, we set the path of the extractor as ```args.pretrain```, 
assuming it is "/trained_2D.pth". Next, we train the detector for single-view prediction.
```
python main.py -d cvcs --variant 2D_SVP --pretrain /trained_2D.pth
```
Samely, when the single-view detector is trained well, assuming it is ```/trained_2D_SVP.pth```. Next, 
we train the final detector.
```
python main.py -d cvcs --variant 2D_SVP_VCW --pretrain /trained_2D_SVP.pth
```
On Wildtrack and MultiviewX, we take the final detector trained on CVCS as the model, then test it with fine-tuning 
and domain-adaptation techniques.


## Pretrained models
You can download the checkpoints at this link.

## Acknowledgement
This work was supported in parts by NSFC (62202312, 62161146005, U21B2023, U2001206), DEGP Innovation Team 
(2022KCXTD025), CityU Strategic Research Grant (7005665), and Shenzhen Science and Technology Program 
(KQTD20210811090044003, RCJC20200714114435012, JCYJ20210324120213036).

## Reference
```
@inproceedings{MVD24,
title={Multi-view People Detection in Large Scenes via Supervised View-wise Contribution Weighting},
author={Qi Zhang and Yunfei Gong and Daijie Chen and Antoni B. Chan and Hui Huang},
booktitle={AAAI Conference on Artificial Intelligence},
pages={7242--7250},
year={2024},
}

```
