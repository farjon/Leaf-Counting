"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from .. import backend
from ..utils import anchors as utils_anchors

import numpy as np


class Anchors(keras.layers.Layer):
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)[:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return backend.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    def __init__(self, mean=None, std=None, *args, **kwargs):
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class ClipBoxes(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())

        x1 = backend.clip_by_value(boxes[:, :, 0], 0, shape[2])
        y1 = backend.clip_by_value(boxes[:, :, 1], 0, shape[1])
        x2 = backend.clip_by_value(boxes[:, :, 2], 0, shape[2])
        y2 = backend.clip_by_value(boxes[:, :, 3], 0, shape[1])

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class SpatialNMS(keras.layers.Layer):
    def __init__(self, kernel_size=(3,3), strides=(1,1), beta=100, *args, **kwargs):
        if not(type(kernel_size)== tuple):
            raise ('kernel size must be tuple and not {}'.format(type(kernel_size)))
        if not(len(kernel_size)== 2):
            raise ('kernel size must have 2 dimensions and not {}'.format(len(kernel_size)))
        if not(type(beta)== int):
            raise ('beta must be int and not {}'.format(type(beta)))
        if not(beta > 0):
            raise ('beta must be graeter than zero')
        if not(type(strides)== tuple):
            raise ('strides must be int and not {}'.format(type(strides)))
        if not(len(strides)== 2):
            raise ('strides must have 2 dimensions and not {}'.format(len(strides)))

        self.kernel_size = kernel_size
        self.beta = beta
        self.strides = strides
        super(SpatialNMS, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        p = inputs
        q = keras.layers.MaxPool2D(pool_size=self.kernel_size, strides=self.strides, padding='same')(p)
        abs_p_sub_q = keras.backend.abs(p-q)
        exp = keras.backend.exp(-abs_p_sub_q * self.beta)
        p2 = keras.layers.multiply([p,exp])
        return p2

    # def compute_output_shape(self, input_shape):
    #     return input_shape

class GlobalSumPooling2D(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(GlobalSumPooling2D, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        return backend.reduce_sum(inputs, [1,2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[3])

class MC_dropout(keras.layers.Layer):
    def __init__(self,level = 0.2, *args, **kwargs):
        self.level = level
        super(MC_dropout, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        #learning_phase = keras.retina_backend.learning_phase()
        #keras.retina_backend.set_learning_phase(1)
        output = keras.backend.dropout(inputs, level=self.level)
        #keras.retina_backend.set_learning_phase(learning_phase)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class StepFunction(keras.layers.Layer):
    def __init__(self, threshold=0.1, *args, **kwargs):
        self.threshold = threshold
        super(StepFunction, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        threshold_factor = keras.backend.ones_like(inputs) * self.threshold
        ones_factor = keras.backend.ones_like(inputs)
        zeros_factor = keras.backend.ones_like(inputs) * 0
        return backend.where(keras.backend.less(inputs, threshold_factor), zeros_factor, ones_factor)

    def compute_output_shape(self, input_shape):
         return input_shape


class MLE_layer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MLE_layer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        _, parameters_dim = inputs.shape.dims
        mone = 0
        mechane = 0
        for param in range(0,parameters_dim,2):
            y = inputs[0, param]
            sig = inputs[0, param + 1]
            p = 1 / keras.backend.exp(sig)
            mone = mone + y*p
            mechane = mechane + p

        return keras.backend.expand_dims(mone/(mechane + keras.backend.epsilon()),axis=0)

    def compute_output_shape(self, input_shape):
         return (1,1)

class SmoothStepFunction1(keras.layers.Layer):
    def __init__(self, threshold=0.8, beta = 15, *args, **kwargs):
        self.threshold = threshold
        self.beta = beta
        super(SmoothStepFunction1, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        threshold_factor = keras.backend.ones_like(inputs) * self.threshold
        sigmoid_input = keras.layers.subtract([(inputs),threshold_factor])* self.beta
        return keras.backend.sigmoid(sigmoid_input)


    # def compute_output_shape(self, input_shape):
    #     return input_shape

    # retina_backend.local_softmax2D(last_layer, self.kernel_size, self.strides, self.beta)
    # def compute_output_shape(self, input_shape):
    #      return (1,1)

class SmoothStepFunction(keras.layers.Layer):
    def __init__(self, threshold=0.4, beta = 1, *args, **kwargs):
        self.threshold = threshold
        self.beta = beta
        super(SmoothStepFunction, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        """
        input: 4D Tensor (b_size, weight, high, f_maps)
        output: 3D Tensor (b_size, sum_of_w_AND_h, f_maps)
        """
        threshold_factor = keras.backend.ones_like(inputs) * self.threshold
        sigmoid_input = keras.layers.subtract([(inputs),threshold_factor])* self.beta
        return keras.backend.sigmoid(sigmoid_input)


    # def compute_output_shape(self, input_shape):
    #     return input_shape

    # retina_backend.local_softmax2D(last_layer, self.kernel_size, self.strides, self.beta)