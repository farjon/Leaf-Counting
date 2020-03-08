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


import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
__package__ = "keras_retinanet.bin"

import keras
from .. import layers

def create_classification_graph(
        inputs,
        num_classes,
        classification_feature_size=256,
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {'kernel_size': 3, 'strides': 1, 'padding': 'same'}
    options_for_middle_output = {'kernel_size': 1, 'strides': 1, 'padding': 'same'}
    outputs = inputs


    middle_outputs = []
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)
        middle_outputs.append(keras.layers.Conv2D(
            filters=num_classes,
            activation='relu',
            name='pyramid_classification_mid_output{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options_for_middle_output,
        )(outputs)
        )

    output_final = keras.layers.Conv2D(
        filters=num_classes,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=keras.initializers.zeros(),
        name='pyramid_classification',
        **options_for_middle_output
    )(outputs)

    output_final = keras.layers.Activation('relu', name='pyramid_classification_relu')(output_final)

    # the output of the model will contain each of the middle_outputs, the last convolution output (for the global pooling)
    # and the output of the final detection sub-model after a 1X1 convolution (which is the density estimation map)
    return middle_outputs + [outputs] + [output_final]


def create_P3_feature(C3, C4, C5, feature_size=256):
    """ Creates the FPN layer P3 on top of the backbone features.

    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        The feature layer P3
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    return P3


def gyf_net(inputs, backbone_layers, num_classes, name='gyf_net'):
    '''
    Construct a detection with regression model on top of a backbone.
    :param inputs:              keras.layers.Input (or list of) for the input to the model.
    :param backbone_layers:     Resnet50 backbone blocks
    :param num_classes:         Number of classes to classify
    :param option:              Can be either 10 or 20, see train_det.py for additional details
    :param name:                Name of the model
    :return:                    A keras.models.Model which takes an image as input and outputs the result from the detection submodel on top of pyramid level P3.
    '''

    C3, C4, C5 = backbone_layers

    P3 = create_P3_feature(C3, C4, C5)

    # create the detection with regression model for keypoint finding and counting
    model_outputs = create_classification_graph(P3, num_classes)
    final_model_output = model_outputs[-1]

    # GAP on last 3x3 conv layer to get 256 feature vector for regression
    det_submodel_last_conv_layer = model_outputs[-2]
    det_submodel_last_conv_layer = keras.layers.GlobalAveragePooling2D(name='256_featurs_for_reg')(det_submodel_last_conv_layer)

    # counting subnetwork including spatial non maxima supression function and smooth step function
    final_output_Step_Function = layers.SmoothStepFunction(threshold=0.4, beta = 1)(final_model_output)
    final_output_LocalNMS = layers.SpatialNMS(kernel_size=(3, 3), strides=(1, 1), beta=100, name='LocalSoftMax')(final_output_Step_Function)
    final_output_Step_Function1 = layers.SmoothStepFunction1(threshold=0.8, beta=15, name='smooth_step_function2')(final_output_LocalNMS)
    final_output_downsampled = layers.GlobalSumPooling2D(name='SumPooling_cls_output')(final_output_Step_Function1)
    # final_output_downsampled will be a the first estimation of the number of leaves (without regression)

    reg_output_downsampled = keras.layers.Concatenate(axis=-1)([det_submodel_last_conv_layer, final_output_downsampled])

    reg_output_downsampled = keras.layers.Dense(1,
                                                name="reg_output",
                                                kernel_initializer=keras.initializers.normal(mean=0.5, stddev=0.1, seed=None),
                                                bias_initializer='zeros'
                                                )(reg_output_downsampled)

    # final output will be the final estimation after regression and each of the middle density estimation maps
    outputs = [reg_output_downsampled] + model_outputs
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def gyf_net_LCC(model, name='gyf_net-LCC', **kwargs):
    '''
    this function is usually used to create a predictive model and not one for training
        if model is none
            Construct a detection with regression model on top of a backbone.
        else
            this function takes model (which is a gyf_net model) and declares its outputs
    :param model: a gyf model
    :param name: Name of the model
    :param kwargs: additional agruments
    :return: a detection with regression model
    '''

    if model is None:
        model = gyf_net(**kwargs)

    regression = model.outputs[0]
    model_outputs = model.outputs[-1]

    outputs = [regression, model_outputs]

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=outputs, name=name)