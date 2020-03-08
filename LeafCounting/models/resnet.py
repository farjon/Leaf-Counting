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

import warnings

import keras
import keras_resnet
import keras_resnet.models
import keras.applications.imagenet_utils
from . import gyf_net_reg, gyf_net_keyPfinder
from . import Backbone


class ResNetBackbone(Backbone):
    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def gyf_net(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_gyf_net(*args, backbone=self.backbone, **kwargs)

    def gyf_net_reg(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_gyf_net('reg', *args, backbone=self.backbone, **kwargs)

    def gyf_net_keyPfinder(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_gyf_net('keyPfinder', *args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return keras.applications.imagenet_utils.get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))


def resnet_gyf_net(type, num_classes, option = 'reg_fpn_p3_p7_mle', backbone='resnet50', inputs=None, modifier=None, do_dropout = None, **kwargs):
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    if type == 'reg':
        gyf_net = gyf_net_reg
    elif type == 'keyPfinder':
        gyf_net = gyf_net_keyPfinder
        return gyf_net.gyf_net(inputs=inputs, backbone_layers=resnet.outputs[1:], num_classes=num_classes, **kwargs)


    # create the full model
    if not (do_dropout is None):
        return gyf_net.gyf_net(inputs=inputs, backbone_layers=resnet.outputs[1:], num_classes=num_classes, option=option, do_dropout=do_dropout, **kwargs)
    else:
        return gyf_net.gyf_net(inputs=inputs, backbone_layers=resnet.outputs[1:], num_classes=num_classes, option=option, **kwargs)
