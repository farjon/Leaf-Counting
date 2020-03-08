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

import tensorflow
import keras


def map_fn(*args, **kwargs):
    return tensorflow.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    return tensorflow.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    return tensorflow.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    return tensorflow.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest' : tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tensorflow.image.ResizeMethod.BICUBIC,
        'area'    : tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.image.resize_images(images, size, methods[method], align_corners)


def non_max_suppression(*args, **kwargs):
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    return tensorflow.where(*args, **kwargs)


def reduce_sum(*args, **kwargs):
    return tensorflow.reduce_sum(*args, **kwargs)

def Slice(*args, **kwargs):
    return tensorflow.slice(*args, **kwargs)


def SparseTensor(*args, **kwargs):
    return tensorflow.SparseTensor(*args, **kwargs)

def cond(*args, **kwargs):
    return tensorflow.cond(*args, **kwargs)

def div(*args, **kwargs):
    return tensorflow.div(*args, **kwargs)
