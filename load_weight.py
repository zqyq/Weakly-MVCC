import os

import h5py
import numpy as np
import torch

translate1 = {'feature_extraction_corr': 'corr_encoder',
             'single_view_feature_extraction': 'base_pt',
              'condfidence_pred_model2': 'confidence_decoder',
              'dist_feature_pred_model': 'distence_extractor',
              'weight_model_pred': 'weight_pred',
              'tranform_view_layer2_1': 'match_CNN'
              }

translate2 = {'decoder': 'img_classifier',}

translate3 = {'tranform_view_layer2_2_1': 'match_CNN',
              'tranform_view_layer2_2_mask_1': 'match_CNN',
              }


def translate_model2(hf):
    useful_info = {}
    key = 'decoder'
    for i in range(8, 18):
        tmp = (i - 8) * 2
        module_name = translate2[key] + '.' + str(tmp)
        layer_name = 'conv' + str(i)
        for layer in hf[key][layer_name].keys():
            tmp_name = module_name + '.' + layer
            tmp_name = tmp_name[:-2]
            if layer[:-2] == 'kernel':
                tmp_name = tmp_name.replace('kernel', 'weight')
                useful_info.update({tmp_name: torch.from_numpy(hf[key][layer_name][layer][:]).permute(3, 2, 0, 1)})
            else:
                useful_info.update({tmp_name: torch.from_numpy(hf[key][layer_name][layer][:])})
    return useful_info

def translate_model3(hf, key):
    useful_info = {}
    for layer in hf[key][key].keys():
        tmp_name = translate3[key] + '.' + str((int(layer[-3:-2]) - 1) * 2) + '.' + layer
        tmp_name = tmp_name[:-3]
        if layer[:-3] == 'kernel':
            tmp_name = tmp_name.replace('kernel', 'weight')
            useful_info.update({tmp_name: torch.from_numpy(hf[key][key][layer][:]).permute(3, 2, 0, 1)})
        else:
            useful_info.update({tmp_name: torch.from_numpy(hf[key][key][layer][:])})
    return useful_info

def translate_model1(hf):
    useful_info = {}
    for key in hf.keys():
        if len(hf[key].keys()) == 0 or key not in translate1:
            continue
        # if (key == 'feature_extraction_corr' or key == 'weight_model_pred' or key == 'single_view_feature_extraction'):
        #     continue
        for i, layer_name in enumerate(hf[key].keys()):
            tmp = i * 2
            if (translate1[key] == 'base_pt' or translate1[key] == 'corr_encoder'):
                if i >= 2 and i < 4:
                    tmp += 1
                if i >= 4:
                    tmp += 2
            module_name = translate1[key] + '.' + str(tmp)
            for layer in hf[key][layer_name].keys():
                if layer_name == 'dense_1' or layer_name == 'dense_2':
                    if layer_name[-1] == '1':
                        module_name = translate1[key] + '.9'
                    else:
                        module_name = translate1[key] + '.10'

                    tmp_name = module_name + '.' + layer
                    tmp_name = tmp_name[:-2]
                    if layer[:-2] == 'kernel':
                        tmp_name = tmp_name.replace('kernel', 'weight')
                        useful_info.update(
                            {tmp_name: torch.from_numpy(hf[key][layer_name][layer][:]).permute(1, 0)})
                    else:
                        useful_info.update({tmp_name: torch.from_numpy(hf[key][layer_name][layer][:])})

                    continue

                tmp_name = module_name + '.' + layer
                tmp_name = tmp_name[:-2]
                if layer[:-2] == 'kernel':
                    tmp_name = tmp_name.replace('kernel', 'weight')
                    useful_info.update({tmp_name: torch.from_numpy(hf[key][layer_name][layer][:]).permute(3, 2, 0, 1)})
                else:
                    useful_info.update({tmp_name: torch.from_numpy(hf[key][layer_name][layer][:])})

    return useful_info

def CVCS_load_weight():
    weight_path = r"/mnt/d/DJ/CF/ pretrain_model/city/372-0.2053-better.h5"
    # base_path = "/mnt/d/DJ/CF/ pretrain_model/origion/Labels_Reproduce_homography2_supervision_counting_FeaDistAtt_rightOutputDim_GD/"

    # for dir in os.listdir(base_path):
    # try:
    # weight_path = os.path.join(base_path, dir)
    useful_infos = {}
    with h5py.File(weight_path, 'r') as hf:
        useful_infos.update(translate_model1(hf))
        for key in hf.keys():
            if key == 'decoder':
                useful_infos.update(translate_model2(hf))
            if key in translate3:
                useful_infos.update(translate_model3(hf, key))

    # weight1_path = r"/mnt/d/DJ/CF/ pretrain_model/origion/2d_pretrain.pth"
    # # weight_path = r"/mnt/d/DJ/CF/ pretrain_model/origion/homo_pre.pth"
    # pretrain_model = torch.load(weight1_path)
    # pretrain_para = pretrain_model['model']
    # extra_info1 = {k: pretrain_para[k] for k in pretrain_para if k.split('.')[0] == 'img_classifier' or k.split('.')[0] == 'base_pt'}
    # # extra_info = {k: pretrain_para[k] for k in pretrain_para if
    # #               k.split('.')[0] == 'corr_encoder' or k.split('.')[0] == 'weight_pred'}
    #
    # weight2_path = r"/mnt/d/DJ/CF/ pretrain_model/origion/homo_pre.pth"
    # # weight_path = r"/mnt/d/DJ/CF/ pretrain_model/origion/homo_pre.pth"
    # pretrain_model = torch.load(weight2_path)
    # pretrain_para = pretrain_model['model']
    # extra_info2 = {k: pretrain_para[k] for k in pretrain_para if
    #                k.split('.')[0] == 'corr_encoder' or k.split('.')[0] == 'weight_pred'}
    #
    # useful_para1 = dict(list(useful_infos.items()) + list(extra_info1.items()) + list(extra_info2.items()))

    checkpoint = {
        'epoch': 1,
        'model': useful_infos,
        # 'model': useful_para1
    }

    torch.save(checkpoint, weight_path.replace('h5', 'pth'))
    # except:
    #     print(dir)

def city_counting_model_split(hf, k):
    useful_infos = {}
    for i, layer_name in enumerate(hf.keys()):
        # if layer_name[:-1] == 'conv' or len(layer_name) == 7:
        if k == 'single_view_feature_extraction' or k == 'weight_model_feature':
            if len(layer_name) == 7 or k == 'weight_model_feature':
                idx = int(layer_name[4]) - 1
            else:
                idx = int(layer_name[-1]) - 1
            tmp = max(0, idx * 2)
            if idx >= 2 and idx < 4:
                tmp += 1
            if idx >= 4:
                tmp += 2
            # if len(layer_name) == 5:
            if k == 'single_view_feature_extraction':
                module_name = 'base_pt' + '.' + str(tmp)
            else:
                module_name = 'corr_encoder' + "." + str(tmp)
        else:
            idx = int(layer_name[7:]) - 1
            tmp = idx * 2
            module_name = 'img_classifier' + '.' + str(tmp)
        for layer in hf[layer_name].keys():
            tmp_name = module_name + '.' + layer[:-2]
            tmp_name = tmp_name.replace('kernel', 'weight')
            if layer[:-2] == 'kernel':
                useful_infos.update({tmp_name: torch.from_numpy(hf[layer_name][layer][:]).permute(3, 2, 0, 1)})
            else:
                useful_infos.update({tmp_name: torch.from_numpy(hf[layer_name][layer][:])})

    return useful_infos


def city_translate(hf, name):
    useful_infos = {}
    for i, layer_name in enumerate(hf.keys()):
        tmp = i * 2
        if name == 'weight_fusion_pred':
            model_name = 'match_CNN' + '.' + str(tmp)
        elif name  == 'condfidence_pred_model2':
            model_name = 'confidence_decoder' + '.' + str(tmp)
        elif name == 'weight_model_pred':
            model_name = 'weight_pred' + '.' + str(tmp)
        else:
            model_name = 'distence_extractor' + '.' + str(tmp)

        for layer in hf[layer_name].keys():
            if layer_name[:-2] == 'dense':
                if layer_name[-1] == '1':
                    model_name = model_name.replace('8', '9')

                tmp_name = model_name + '.' + layer[:-2]
                tmp_name = tmp_name.replace('kernel', 'weight')
                if layer[:-2] == 'kernel':
                    useful_infos.update({tmp_name: torch.from_numpy(hf[layer_name][layer][:]).permute(1, 0)})
                else:
                    useful_infos.update({tmp_name: torch.from_numpy(hf[layer_name][layer][:])})

            else:
                tmp_name = model_name + '.' + layer[:-2]
                tmp_name = tmp_name.replace('kernel', 'weight')
                if layer[:-2] == 'kernel':
                    useful_infos.update({tmp_name: torch.from_numpy(hf[layer_name][layer][:]).permute(3, 2, 0, 1)})
                else:
                    useful_infos.update({tmp_name: torch.from_numpy(hf[layer_name][layer][:])})
    return useful_infos

def City_load_weight():
    weight_path = r"/mnt/d/DJ/CF/ pretrain_model/pets/00-24096427.9418-better.h5"
    useful_infos = {}
    with h5py.File(weight_path, 'r') as hf:
        for key in hf.keys():
            if len(hf[key].keys()) == 0:
                continue
            if key == 'counting_model' or key == 'weight_model_feature' or key == 'single_view_feature_extraction':
                useful_infos.update(city_counting_model_split(hf[key], key))
            # elif key == 'single_view_feature_extraction':
            #     useful_infos.update(translate_model1(hf))
            elif key == 'decoder':
                useful_infos.update(translate_model2(hf))
            else:
                useful_infos.update(city_translate(hf[key], key))

    checkpoint = {
        'epoch': 1,
        'model': useful_infos,
        # 'model': useful_para1
    }

    torch.save(checkpoint, weight_path.replace('h5', 'pth'))

if __name__ == '__main__':
    count_result = "/mnt/d/DJ/CF/ pretrain_model/counting_num_test_pred_nae"
    with h5py.File(count_result, 'r') as hf:
        data = hf['mae_GP']
    City_load_weight()
    pass