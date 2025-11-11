# from keras.layers.core import Layer
import torch as tf
import numpy as np
import cv2

# import matplotlib.pyplot as plt

# from keras.engine import InputSpec

import torch.nn as nn


class UpSampling_layer(nn.Module):

    def __init__(self,
                 size=[128, 128],
                 **kwargs):
        super(UpSampling_layer, self).__init__()

        #self.scale = scale
        self.size = size
        #self.view = view


    def build(self, input_shape):
        super(UpSampling_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        size = self.size
        feature = input_shape
        return (int(feature[0]),
                int(size[0]),
                int(size[1]),
                int(feature[3]))

    def call(self, x):
        size = self.size
        height = size[0]
        width = size[1]

        x = x[0]

        feature_UpSampled = tf.image.resize_bilinear(x, size)

        return feature_UpSampled



class Sum_layer(Layer):

    def __init__(self,
                 batch_size,
                 view_size,
                 patch_num = 1,
                 **kwargs):
        self.batch_size = batch_size
        self.view_size = view_size
        self.patch_num = patch_num

        super(Sum_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sum_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        batch_size = self.batch_size
        patch_num = self.patch_num

        return (batch_size, patch_num,
                input_shape[2],
                input_shape[3],
                input_shape[4])

    def call(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        feature_view_pooled = tf.reshape(x, (batch_size, view_size, patch_num,
                                             x.shape[2].value, x.shape[3].value, x.shape[4].value))
        feature_view_pooled = tf.reduce_sum(feature_view_pooled, axis=1, keep_dims=False)
        # feature_view_pooled = tf.reshape(feature_view_pooled, (batch_size*patch_num,
        #                                      x.shape[2].value, x.shape[3].value, x.shape[4].value))

        # # feature_view_pooled = tf.placeholder(tf.float32, [1, x.shape[1], x.shape[2], x.shape[3]])
        # feature_view_pooled = tf.zeros([1, x.shape[1], x.shape[2], x.shape[3]])
        #
        # for i in range(batch_size*patch_num):
        #     x_i = tf.reduce_max(x[i*view_size:(i+1)*view_size], axis=0, keep_dims=True)
        #     feature_view_pooled = tf.concat([feature_view_pooled, x_i], axis=0)
        #
        # feature_view_pooled = feature_view_pooled[1:]

        return feature_view_pooled




class camera_sel_fusion_layer_rbm(Layer):
    def __init__(self,
                 batch_size=1,
                 view_size=5,
                 patch_num = 5,
                 **kwargs):
        self.patch_num = patch_num
        self.view_size = view_size
        self.batch_size = batch_size

        super(camera_sel_fusion_layer_rbm, self).__init__(**kwargs)

    def build(self, input_shape):
        print('No trainable weights for camera sel layer.')

    def compute_output_shape(self, input_shape):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        return (batch_size*view_size, patch_num,
                 input_shape[1],
                 input_shape[2],
                 input_shape[3]*256)

    def call(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        b_v_p_size = x.shape[0].value # actually, it's b*v*p
        height = x.shape[1].value
        width = x.shape[2].value
        num_channels = x.shape[3].value

        x = tf.reshape(x, (batch_size, view_size, patch_num, height, width, num_channels))
        x = tf.transpose(x, (0, 2, 1, 3, 4, 5))
        x = tf.reshape(x, (batch_size*patch_num, view_size, height, width, num_channels))

        x_clip = tf.clip_by_value(x, 0, 1)

        x_clip2 = (1-x_clip)*1e8

        x_e8 = x + x_clip2
        x_min = tf.reduce_min(x_e8, axis=1, keep_dims=True)
        x_min_tile = tf.tile(x_min, (1, view_size, 1, 1, 1))

        x_sum = tf.reduce_sum(x, axis=1, keep_dims=True)
        x_sum_clip = tf.clip_by_value(x_sum, 0, 1)
        x_sum_clip2 = 1 - x_sum_clip

        x_dist = -(tf.square(x_e8 - x_min_tile)/(100)) # x_min_tile*10, 200, 100
        x_dist2 = tf.exp(x_dist)
        x_dist2_mask = tf.multiply(x_dist2, x_clip)



        x_dist2_mask_sum = tf.reduce_sum(x_dist2_mask, axis=1, keep_dims=True)
        x_dist2_mask_sum2 = tf.tile(x_dist2_mask_sum+x_sum_clip2, (1, view_size, 1, 1, 1))

        x_dist2_mask_sum2_softmax = tf.divide(x_dist2_mask, x_dist2_mask_sum2)
        # x_dist2_mask/x_dist2_mask_sum2
        #tf.nn.softmax(x_dist2_mask_sum2, axis=1)

        x_dist2_mask_sum2_softmax_mask = tf.multiply(x_dist2_mask_sum2_softmax, x_clip)

        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size, patch_num, view_size, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.transpose(x_dist2_mask_sum2_softmax_mask, (0, 2, 1, 3, 4, 5))
        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size*view_size, patch_num, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.tile(x_dist2_mask_sum2_softmax_mask, [1, 1, 1, 1, 256])

        return x_dist2_mask_sum2_softmax_mask #output_mask


class camera_sel_fusion_layer_rbm2(Layer):
    def __init__(self,
                 batch_size=1,
                 view_size=5,
                 patch_num = 5,
                 **kwargs):
        self.patch_num = patch_num
        self.view_size = view_size
        self.batch_size = batch_size

        super(camera_sel_fusion_layer_rbm2, self).__init__(**kwargs)

    def build(self, input_shape):
        print('No trainable weights for camera sel layer.')

    def compute_output_shape(self, input_shape):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        return (batch_size*view_size, patch_num,
                 input_shape[1],
                 input_shape[2],
                 input_shape[3]*256)

    def call(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        b_v_p_size = x.shape[0].value # actually, it's b*v*p
        height = x.shape[1].value
        width = x.shape[2].value
        num_channels = x.shape[3].value

        x = tf.reshape(x, (batch_size, view_size, patch_num, height, width, num_channels))
        x = tf.transpose(x, (0, 2, 1, 3, 4, 5))
        x = tf.reshape(x, (batch_size*patch_num, view_size, height, width, num_channels))

        x_clip = tf.clip_by_value(x, 0, 1)

        x_clip2 = (1-x_clip)*1e8

        x_e8 = x + x_clip2
        x_e8 = tf.log(x_e8)

        x_min = tf.reduce_min(x_e8, axis=1, keep_dims=True)
        x_min_tile = tf.tile(x_min, (1, view_size, 1, 1, 1))

        x_sum = tf.reduce_sum(x, axis=1, keep_dims=True)
        x_sum_clip = tf.clip_by_value(x_sum, 0, 1)
        x_sum_clip2 = 1 - x_sum_clip

        x_dist = -(tf.square(x_e8 - x_min_tile)/(1)) # x_min_tile*10, 200, 100
        x_dist2 = tf.exp(x_dist)
        x_dist2_mask = tf.multiply(x_dist2, x_clip)

        x_dist2_mask_sum = tf.reduce_sum(x_dist2_mask, axis=1, keep_dims=True)
        x_dist2_mask_sum2 = tf.tile(x_dist2_mask_sum+x_sum_clip2, (1, view_size, 1, 1, 1))

        x_dist2_mask_sum2_softmax = tf.divide(x_dist2_mask, x_dist2_mask_sum2)
        # x_dist2_mask/x_dist2_mask_sum2
        #tf.nn.softmax(x_dist2_mask_sum2, axis=1)

        x_dist2_mask_sum2_softmax_mask = tf.multiply(x_dist2_mask_sum2_softmax, x_clip)

        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size, patch_num, view_size, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.transpose(x_dist2_mask_sum2_softmax_mask, (0, 2, 1, 3, 4, 5))
        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size*view_size, patch_num, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.tile(x_dist2_mask_sum2_softmax_mask, [1, 1, 1, 1, 256])

        return x_dist2_mask_sum2_softmax_mask #output_mask



class camera_sel_fusion_layer_rbm2_noSoftmax(Layer):
    def __init__(self,
                 batch_size=1,
                 view_size=5,
                 patch_num = 5,
                 **kwargs):
        self.patch_num = patch_num
        self.view_size = view_size
        self.batch_size = batch_size

        super(camera_sel_fusion_layer_rbm2_noSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        print('No trainable weights for camera sel layer.')

    def compute_output_shape(self, input_shape):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        return (batch_size*view_size, patch_num,
                 input_shape[1],
                 input_shape[2],
                 input_shape[3]*256)

    def call(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        b_v_p_size = x.shape[0].value # actually, it's b*v*p
        height = x.shape[1].value
        width = x.shape[2].value
        num_channels = x.shape[3].value

        x = tf.reshape(x, (batch_size, view_size, patch_num, height, width, num_channels))
        x = tf.transpose(x, (0, 2, 1, 3, 4, 5))
        x = tf.reshape(x, (batch_size*patch_num, view_size, height, width, num_channels))

        x_clip = tf.clip_by_value(x, 0, 1)

        x_clip2 = (1-x_clip)*1e8

        x_e8 = x + x_clip2
        x_e8 = tf.log(x_e8)

        x_min = tf.reduce_min(x_e8, axis=1, keep_dims=True)
        x_min_tile = tf.tile(x_min, (1, view_size, 1, 1, 1))

        x_sum = tf.reduce_sum(x, axis=1, keep_dims=True)
        x_sum_clip = tf.clip_by_value(x_sum, 0, 1)
        x_sum_clip2 = 1 - x_sum_clip

        x_dist = -(tf.square(x_e8 - x_min_tile)/(1)) # x_min_tile*10, 200, 100
        x_dist2 = tf.exp(x_dist)
        x_dist2_mask = tf.multiply(x_dist2, x_clip)

        x_dist2_mask_sum = tf.reduce_sum(x_dist2_mask, axis=1, keep_dims=True)
        x_dist2_mask_sum2 = tf.tile(x_dist2_mask_sum+x_sum_clip2, (1, view_size, 1, 1, 1))

        # x_dist2_mask_sum2_softmax = tf.divide(x_dist2_mask, x_dist2_mask_sum2)
        # x_dist2_mask_sum2_softmax_mask = tf.multiply(x_dist2_mask_sum2_softmax, x_clip)
        x_dist2_mask_sum2_softmax_mask = x_dist2_mask

        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size, patch_num, view_size, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.transpose(x_dist2_mask_sum2_softmax_mask, (0, 2, 1, 3, 4, 5))
        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size*view_size, patch_num, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.tile(x_dist2_mask_sum2_softmax_mask, [1, 1, 1, 1, 256])

        return x_dist2_mask_sum2_softmax_mask #output_mask




class camera_sel_fusion_layer_rbm_one2(Layer):
    def __init__(self,
                 batch_size=1,
                 view_size=5,
                 patch_num = 5,
                 **kwargs):
        self.patch_num = patch_num
        self.view_size = view_size
        self.batch_size = batch_size

        super(camera_sel_fusion_layer_rbm_one2, self).__init__(**kwargs)

    def build(self, input_shape):
        print('No trainable weights for camera sel layer.')

    def compute_output_shape(self, input_shape):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        return (batch_size*view_size, patch_num,
                 input_shape[1],
                 input_shape[2],
                 input_shape[3]*256)

    def call(self, x):
        batch_size = self.batch_size
        view_size = self.view_size
        patch_num = self.patch_num

        b_v_p_size = x.shape[0].value # actually, it's b*v*p
        height = x.shape[1].value
        width = x.shape[2].value
        num_channels = x.shape[3].value

        x = tf.reshape(x, (batch_size, view_size, patch_num, height, width, num_channels))
        x = tf.transpose(x, (0, 2, 1, 3, 4, 5))
        x = tf.reshape(x, (batch_size*patch_num, view_size, height, width, num_channels))

        x_clip = tf.clip_by_value(x, 0, 1)

        x_clip2 = (1-x_clip)*1e8

        x_e8 = x + x_clip2
        # x_e8 = tf.log(x_e8) #/ tf.log(1.5)

        x_min = tf.reduce_min(x_e8, axis=1, keep_dims=True)
        x_min_tile = tf.tile(x_min, (1, view_size, 1, 1, 1))

        x_sum = tf.reduce_sum(x, axis=1, keep_dims=True)
        x_sum_clip = tf.clip_by_value(x_sum, 0, 1)
        x_sum_clip2 = 1 - x_sum_clip

        # x_dist = -(tf.abs(x_e8 - x_min_tile)/(25)) # x_min_tile*10, 200 square
        x_dist = -(tf.abs(x_e8 - x_min_tile) / (x_min_tile))  # x_min_tile*10, 200 square

        x_dist2 = tf.exp(x_dist)

        x_dist2_mask = tf.multiply(x_dist2, x_clip)

        x_dist2_mask_sum = tf.reduce_sum(x_dist2_mask, axis=1, keep_dims=True)
        x_dist2_mask_sum2 = tf.tile(x_dist2_mask_sum + x_sum_clip2, (1, view_size, 1, 1, 1))

        x_dist2_mask_sum2_softmax = tf.divide(x_dist2_mask, 1)  # x_dist2_mask_sum2
        # x_dist2_mask/x_dist2_mask_sum2
        # tf.nn.softmax(x_dist2_mask_sum2, axis=1)

        x_dist2_mask_sum2_softmax_mask = tf.multiply(x_dist2_mask_sum2_softmax, x_clip)

        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size, patch_num, view_size, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.transpose(x_dist2_mask_sum2_softmax_mask, (0, 2, 1, 3, 4, 5))
        x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
                                                    (batch_size*view_size, patch_num, height, width, num_channels))
        x_dist2_mask_sum2_softmax_mask = tf.tile(x_dist2_mask_sum2_softmax_mask, [1, 1, 1, 1, 256])

        return x_dist2_mask_sum2_softmax_mask #output_mask


# class camera_sel_fusion_layer_rbm_one2(Layer):
#     def __init__(self,
#                  view_size=5,
#                  batch_size = 1,
#                  **kwargs):
#         # self.scale = scale
#         self.view_size = view_size
#         self.batch_size = batch_size
#
#         super(camera_sel_fusion_layer_rbm_one2, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         print('No trainable weights for camera sel layer.')
#
#     def compute_output_shape(self, input_shape):
#         batch_size = self.batch_size
#         view_size = self.view_size
#
#         return (batch_size*view_size, # input_shape[0],
#                  input_shape[1],
#                  input_shape[2],
#                  input_shape[3]*256)
#
#     def call(self, x):
#         batch_size = self.batch_size
#         view_number = self.view_size
#
#         b_v_size = x.shape[0].value  # actually, it's b*v
#         height = x.shape[1].value
#         width = x.shape[2].value
#         num_channels = x.shape[3].value
#
#         x = tf.reshape(x, (batch_size, view_number, height, width, num_channels))
#         x_clip = tf.clip_by_value(x, 0, 1)
#
#         x_clip2 = (1 - x_clip) * 1e8
#
#         x_e8 = x + x_clip2
#         # x_e8 = tf.log(x_e8) #/ tf.log(1.5)
#
#         x_min = tf.reduce_min(x_e8, axis=1, keep_dims=True)
#         x_min_tile = tf.tile(x_min, (1, view_number, 1, 1, 1))
#
#         x_sum = tf.reduce_sum(x, axis=1, keep_dims=True)
#         x_sum_clip = tf.clip_by_value(x_sum, 0, 1)
#         x_sum_clip2 = 1 - x_sum_clip
#
#         # x_dist = -(tf.abs(x_e8 - x_min_tile)/(25)) # x_min_tile*10, 200 square
#         x_dist = -(tf.abs(x_e8 - x_min_tile)/(x_min_tile)) # x_min_tile*10, 200 square
#
#         x_dist2 = tf.exp(x_dist)
#
#         x_dist2_mask = tf.multiply(x_dist2, x_clip)
#
#         x_dist2_mask_sum = tf.reduce_sum(x_dist2_mask, axis=1, keep_dims=True)
#         x_dist2_mask_sum2 = tf.tile(x_dist2_mask_sum + x_sum_clip2, (1, view_number, 1, 1, 1))
#
#         x_dist2_mask_sum2_softmax = tf.divide(x_dist2_mask, 1) #x_dist2_mask_sum2
#         # x_dist2_mask/x_dist2_mask_sum2
#         # tf.nn.softmax(x_dist2_mask_sum2, axis=1)
#
#         x_dist2_mask_sum2_softmax_mask = tf.multiply(x_dist2_mask_sum2_softmax, x_clip)
#         x_dist2_mask_sum2_softmax_mask = tf.reshape(x_dist2_mask_sum2_softmax_mask,
#                                                     (batch_size * view_number, height, width, num_channels))
#
#         x_dist2_mask_sum2_softmax_mask = tf.tile(x_dist2_mask_sum2_softmax_mask, [1, 1, 1, 256])
#
#         return x_dist2_mask_sum2_softmax_mask


class Generate_random_cropping_layer(Layer):

    def __init__(self,
                 cropped_size,
                 patch_num = 1,
                 **kwargs):
        self.cropped_size = cropped_size
        self.patch_num = patch_num

        super(Generate_random_cropping_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(Generate_random_cropping_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        patch_num = self.patch_num
        feature = input_shape
        return (int(feature[0]),
                int(patch_num),
                int(2))

    def call(self, x):
        patch_num = self.patch_num
        cropped_size = self.cropped_size

        input_wld_map_paras = x
        batch_size = input_wld_map_paras.shape[0].value

        hw_random_all = tf.zeros([1, patch_num, 2])
        for i in range(batch_size):

            h_actual, w_actual = input_wld_map_paras[i, 4], input_wld_map_paras[i, 5]

            hw_random_i = tf.zeros([1, 2])
            for j in range(patch_num):
                h_random = tf.random.uniform(shape=(patch_num, 1),
                                             minval=0, maxval=h_actual-cropped_size[0],
                                             dtype=tf.dtypes.int32,
                                             seed=None, name=None)
                w_random = tf.random.uniform(shape=(patch_num, 1),
                                             minval=0, maxval=w_actual-cropped_size[1],
                                             dtype=tf.dtypes.int32,
                                             seed=None, name=None)
                hw_random = tf.concat(h_random, w_random, axis=-1)
                hw_random_i = tf.concat(hw_random, w_random, axis=0)
            hw_random_i = hw_random_i[1:, :]
            hw_random_all = tf.concat(hw_random_all, hw_random_i, axis=0)
        hw_random_all = hw_random_all[1:, :]

        return hw_random_all



class Cropping_layer(Layer):

    def __init__(self,
                 input_size,
                 cropped_size,
                 resize,
                 batch_size,
                 **kwargs):
        self.input_size = input_size
        self.cropped_size = cropped_size
        self.resize = resize
        self.batch_size = batch_size
        super(Cropping_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        if input_shape[0]==None:
            b=self.batch_size
        else:
            b = input_shape[0]
        super(Cropping_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        cropped_size = self.cropped_size # h, w
        input_size = self.input_size
        batch_size = self.batch_size

        # feature = input_shape[0]
        # if feature.shape[0].value==None:
        #     feature.set_shape((batch_size, input_size[0], input_size[1], 1))

        return ((batch_size, cropped_size[0], cropped_size[1], 1))

    def call(self, x):
        cropped_size = self.cropped_size # h, w
        input_size = self.input_size
        batch_size = self.batch_size

        y_true, input_wld_map_paras = x
        input_wld_map_paras = input_wld_map_paras[0]

        if y_true.shape[0].value==None:
            y_true.set_shape((batch_size, input_size[0], input_size[1], 1))
            # input_wld_map_paras.set_shape((1, 10))

        # define wld map paras
        s, r, w_max, h_max, h, w, d_delta, d_mean, w_min, h_min = input_wld_map_paras[0], input_wld_map_paras[1], input_wld_map_paras[2], \
                                                          input_wld_map_paras[3], input_wld_map_paras[4], input_wld_map_paras[5], \
                                                          input_wld_map_paras[6], input_wld_map_paras[7], input_wld_map_paras[8], \
                                                          input_wld_map_paras[9]
        r = self.resize
        a = int(r/2)
        b = int(r/2)

        # actual size:
        w_actual = tf.cast((w_max-w_min)*r + 2*a, tf.int32)
        h_actual = tf.cast((h_max-h_min)*r + 2*b, tf.int32)

        h_actual = tf.maximum(h_actual, cropped_size[0])
        w_actual = tf.maximum(w_actual, cropped_size[1])

        output_cropped = y_true[:, :h_actual, :w_actual, :]

        size = [batch_size, cropped_size[0], cropped_size[1], 1]
        output_cropped2 = tf.random_crop(output_cropped, size=size, seed=1)

        return output_cropped2



class Cropping_layer2(Layer):

    def __init__(self,
                 input_size,
                 cropped_size,
                 resize,
                 batch_size,
                 patch_num = 5,
                 **kwargs):
        self.input_size = input_size
        self.cropped_size = cropped_size
        self.resize = resize
        self.batch_size = batch_size

        self.patch_num = patch_num

        super(Cropping_layer2, self).__init__(**kwargs)


    def build(self, input_shape):
        if input_shape[0]==None:
            b=self.batch_size
        else:
            b = input_shape[0]
        super(Cropping_layer2, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        cropped_size = self.cropped_size # h, w
        input_size = self.input_size
        batch_size = self.batch_size

        patch_num = self.patch_num

        # feature = input_shape[0]
        # if feature.shape[0].value==None:
        #     feature.set_shape((batch_size, input_size[0], input_size[1], 1))

        return ((batch_size*patch_num, cropped_size[0], cropped_size[1], 1))

    def call(self, x):
        cropped_size = self.cropped_size # h, w
        input_size = self.input_size
        batch_size = self.batch_size

        patch_num = self.patch_num

        y_true, input_wld_map_paras = x
        input_wld_map_paras = input_wld_map_paras[0]

        if y_true.shape[0].value==None:
            y_true.set_shape((batch_size, input_size[0], input_size[1], 1))
            # input_wld_map_paras.set_shape((1, 10))

        # define wld map paras
        s, r, w_max, h_max, h, w, d_delta, d_mean, w_min, h_min = input_wld_map_paras[0], input_wld_map_paras[1], input_wld_map_paras[2], \
                                                          input_wld_map_paras[3], input_wld_map_paras[4], input_wld_map_paras[5], \
                                                          input_wld_map_paras[6], input_wld_map_paras[7], input_wld_map_paras[8], \
                                                          input_wld_map_paras[9]
        r = self.resize
        a = int(r/2)
        b = int(r/2)

        # actual size:
        w_actual = tf.cast((w_max-w_min)*r + 2*a, tf.int32)
        h_actual = tf.cast((h_max-h_min)*r + 2*b, tf.int32)

        h_actual = tf.maximum(h_actual, cropped_size[0])
        w_actual = tf.maximum(w_actual, cropped_size[1])

        output_cropped = y_true[:, :h_actual, :w_actual, :]

        # size = [batch_size, cropped_size[0], cropped_size[1], 1]
        # output_cropped2 = tf.random_crop(output_cropped, size=size, seed=1)

        size = [batch_size, cropped_size[0], cropped_size[1], 1]

        seed_range = range(patch_num)
        output_cropped2 = tf.zeros([batch_size, cropped_size[0], cropped_size[1], 1])
        for i in seed_range:
            output_cropped2_i = tf.random_crop(output_cropped, size=size, seed=i)
            output_cropped2 = tf.concat([output_cropped2, output_cropped2_i], axis=0)

        output_cropped2 = output_cropped2[1:]

        return output_cropped2