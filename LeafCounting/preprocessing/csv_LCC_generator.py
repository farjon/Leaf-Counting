"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
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

from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
    resize_image_320,
    read_image_bgr,
    read_image_gray_scale
)
from ..utils.transform import transform_ab


import keras
import numpy as np
from PIL import Image
from six import raise_from

import random
import threading

import csv
import sys
import os.path
import cv2
import seaborn
import matplotlib.pyplot as plt


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations_NOL(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, num_of_leafs = row[:2]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file, num_of_leafs\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (num_of_leafs) == (''):
            raise(ValueError('image {}: doesnt contain label\''.format(img_file)), None)

        # Check that the bounding box is valid.
        if int(num_of_leafs) <= 0:
            raise ValueError('num_of_leafs must be higher than 0 but is {}'.format(num_of_leafs))

        result[img_file].append({'num_of_leafs': num_of_leafs, 'class': 'leafs'})
    return result



def create_gausian_mask(center_point, nCols, nRows, q = 99, radius = (5,5)):
    '''
    create_gausian_mask creates a gaussian mask to be used as GT annotations for the detection-based counter
    :param center_point:
    :param nCols:
    :param nRows:
    :param q:
    :param s:
    :param radius:
    :return:
    '''
    s = 3
    if (s>=radius[0]):
        s=1
    x = np.tile(range(nCols), (nRows,1))
    y = np.tile(np.reshape(range(nRows),(nRows,1)),(1,nCols))

    x2 = (((x - round(center_point[0]))*s) / radius[0]) ** 2
    y2 = (((y - round(center_point[1]))*s) / radius[1]) ** 2

    p = np.exp(-0.5 * (x2 + y2))

    p[np.where(p < np.percentile(p, q))] = 0

    p = p / np.max(p)
    if not np.isfinite(p).all() or not np.isfinite(p).all():
        print('divide by zero')
    return p


def image_output_shape(image_shape,pyramid_level = 3):
    return (np.array(image_shape[:2]) + 2 ** pyramid_level - 1) // (2 ** pyramid_level)


def _read_annotations_leafs_locations(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x, y = row[:3]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x,y\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x, y) == ('', ''):
            raise (ValueError('image {}: doesnt contain label\''.format(img_file)), None)

        x1 = _parse(x, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y, int, 'line {}: malformed y1: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x1 < 0:
            raise ValueError('line {}: x ({}) must be higher than 0 ({})'.format(line, x))
        if y1 < 0:
            raise ValueError('line {}: y ({}) must be higher than 0 ({})'.format(line, y))

        result[img_file].append({'x': x, 'y': y, 'class': 'leafs'})
    return result

def images_ratios(image_shape, output_shape):
    return output_shape / np.array(image_shape[:2])


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVLCCGenerator(object):
    def __init__(
        self,
        csv_leaf_number_file,
        csv_leaf_location_file,
        option = 20,
        base_dir=None,
        transform_generator=None,
        batch_size=1,
        group_method='random',  # one of 'none', 'random', 'ratio'
        shuffle_groups=False,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        epoch=None,
        **kwargs
    ):
        self.csv_leaf_number_file = csv_leaf_number_file
        self.csv_leaf_location_file = csv_leaf_location_file
        self.option = option
        self.rbg_images_names = []
        self.centers_images_names = []
        self.image_data_leaf_number  = {}
        self.image_data_leaf_location  = {}
        self.base_dir    = base_dir
        self.epoch       = epoch


        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_leaf_number_file)

        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()

        self.labels = {'0': 'leafs'}
        self.classes = {'leafs' : 0}

        if csv_leaf_number_file:
            # csv with img_path, num_of_leafs
            try:
                with _open_for_csv(csv_leaf_number_file) as file:
                    self.image_data_leaf_number = _read_annotations_NOL(csv.reader(file, delimiter=','))
            except ValueError as e:
                raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_leaf_number_file, e)), None)
            rbg_images_names = list(self.image_data_leaf_number.keys())
            self.rbg_images_names = rbg_images_names

            # csv with img_path, x, y
            try:
                with _open_for_csv(csv_leaf_location_file) as file:
                    self.image_data_leaf_location = _read_annotations_leafs_locations(csv.reader(file, delimiter=','))
            except ValueError as e:
                raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_leaf_location_file, e)), None)
            self.centers_images_names =[x.replace('rgb', 'centers') for x in rbg_images_names]
            assert set(list(self.image_data_leaf_location.keys()))== set(self.centers_images_names) , 'there are some missing centers annotations'
        else:
            rbg_images_names = os.listdir(self.base_dir)
            rbg_images_names_a =[]

            for im in rbg_images_names:
                if 'CVPPP' in base_dir:
                    if im.split('_')[-1] == 'rgb.png':
                        rbg_images_names_a.append(im)
                else:
                    rbg_images_names_a.append(im)
            self.rbg_images_names = rbg_images_names_a

        self.group_index = 0
        self.lock        = threading.Lock()
        self.group_images()

    def get_epoch(self):
        return self.epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_option(self):
        return self.option

    def size(self):
        return len(self.rbg_images_names)

    def get_csv_leaf_number_file(self):
        return self.csv_leaf_number_file

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path_rgb(self, image_index):
        return os.path.join(self.base_dir, self.rbg_images_names[image_index])

    def image_path_centers(self, image_index):
        return os.path.join(self.base_dir, self.centers_images_names[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.image_path_rgb(image_index))
        return float(image.width) / float(image.height)

    def load_image_rgb(self, image_index):
        return read_image_bgr(self.image_path_rgb(image_index))

    def load_image_centers(self, image_index):
        return read_image_bgr(self.image_path_centers(image_index))

    def load_image_byName(self, image_name):
        return read_image_bgr(os.path.join(self.base_dir, image_name))

    def load_annotations_num_of_leaves(self, image_index):
        path   = self.rbg_images_names[image_index]
        annots = self.image_data_leaf_number[path]
        counts  = np.zeros((len(annots), 2))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            counts[idx, 0] = float(annot['num_of_leafs'])
            counts[idx, 1] = self.name_to_label(class_name)

        return counts

    def compute_keypoints_targets(self, image_shape, annotations_leaves_centers,):
        output_shape = image_output_shape(image_shape)
        image_ratio = images_ratios(image_shape, output_shape)
        annotations_leaves_centers[:, :2] = annotations_leaves_centers[:, :2] * image_ratio

        annotations = np.zeros(output_shape)
        for i in range(annotations_leaves_centers.shape[0]):

            gaussian_map = create_gausian_mask(annotations_leaves_centers[i, :2], output_shape[1],output_shape[0])
            # each center point in the GT will be 1 in the annotation map
            annotations = np.maximum(annotations, gaussian_map)

        return annotations

    def get_annotations_byName(self):
        annotations_dict = {}
        with open(self.csv_leaf_number_file) as csvfile:
            readCSV = csv.reader(csvfile)
            for row in readCSV:
                annotations_dict[row[0]] = int(row[1])
        return (annotations_dict)

    def get_leaf_coord_byName(self):
        coordinates_dict = {}
        with open(self.csv_leaf_location_file) as csvfile:
            readCSV = csv.reader(csvfile)
            for row in readCSV:
                if len(coordinates_dict) == 0:
                    coordinates_dict[row[0]] = []
                    coordinates_dict[row[0]].append([int(row[1]),int(row[2])])
                else:
                    if row[0] in coordinates_dict.keys():
                        coordinates_dict[row[0]].append([int(row[1]), int(row[2])])
                    else:
                        coordinates_dict[row[0]]=[]
                        coordinates_dict[row[0]].append([int(row[1]), int(row[2])])

        return (coordinates_dict)


    def load_annotations_group_num_of_leaves(self, group):
        return [self.load_annotations_num_of_leaves(image_index) for image_index in group]

    def load_annotations_leaves_centers(self, image_index):
        path   = self.centers_images_names[image_index]
        annots = self.image_data_leaf_location[path]
        centers  = np.zeros((len(annots), 3))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            centers[idx, 0] = float(annot['x'])
            centers[idx, 1] = float(annot['y'])
            centers[idx, 2] = self.name_to_label(class_name)

        return centers

    def load_annotations_group_leaves_center(self, group):
        return [[self.load_annotations_leaves_centers(image_index) for image_index in group]]

    def filter_annotations(self, image_group, annotations_group_leaves_center, annotations_group_num_of_leaves, group):
        annotations_centers = annotations_group_leaves_center[0]
        annotations_num_of_leaves = annotations_group_num_of_leaves[0]
        # test all annotations
        for index, (image, annotation_centers, annotation_n_leaves) in enumerate(zip(image_group, annotations_centers,annotations_num_of_leaves)):
            assert isinstance(annotation_centers, np.ndarray), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotation_centers))
            assert annotation_centers.shape[0]==int(annotation_n_leaves[0]), 'the number of centers is not equal in the two annotations type ({},{}), image: {}'.format(str(annotation_centers.shape[0]),str(annotation_n_leaves[0]),str(self.rbg_images_names[group[index]]))

            # test x < 0 | y < 0 x >= image.shape[1] | y >= image.shape[0]
            # invalid_indices = np.where(
            #     (np.array([annotation_centers[i][0] for i in range(len(annotation_centers[:]))]) < 0) |
            #     (np.array([annotation_centers[i][1] for i in range(len(annotation_centers[:]))]) < 0) |
            #     (np.array([annotation_centers[i][1] for i in range(len(annotation_centers[:]))]) > image.shape[1])|
            #     (np.array([annotation_centers[i][0] for i in range(len(annotation_centers[:]))]) > image.shape[0])
            # )[0]
            #
            # # warn the user so that he can manually delete the false annotation!
            # if len(invalid_indices):
            #     warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
            #         group[index],
            #         image.shape,
            #         [annotation_centers[invalid_index, :] for invalid_index in invalid_indices]
            #     ))
            #     print('Mannualy delete the false annotations!')
            #     raise ()

        return image_group, annotations_group_leaves_center, annotations_group_num_of_leaves

    def load_image_group(self, group):
        return [self.load_image_rgb(image_index) for image_index in group]

    def load_centers_image_group(self, group):
        return [self.load_image_centers(image_index) for image_index in group]

    def random_transform_rbg_centers_images(self, rgb_image, annotations):
        random_success_flag = False
        # randomly transform both image and annotations
        if self.transform_generator:
            while(random_success_flag == False):
                transformation_to_apply = next(self.transform_generator)
                transform_rgb = adjust_transform_for_image(transformation_to_apply, rgb_image, self.transform_parameters.relative_translation)
                res_rgb_image     = apply_transform(transform_rgb, rgb_image, self.transform_parameters)

                annotations_a = annotations.copy()
                for index in range(annotations_a.shape[0]):
                    annotations_a[index, :2] = transform_ab(transform_rgb, annotations_a[index, :2])
                random_success_flag = True
                assert annotations.shape[0]==annotations_a.shape[0], 'there is some buge in the augmantation procses'
                # check if annotations_a contains negative values
                for row in annotations_a:
                    for ele in row:
                        if ele < 0:
                            random_success_flag = False
            return res_rgb_image, annotations_a
        else:
            return rgb_image, annotations

    def resize_image(self, image):
        return resize_image(image)

    def resize_image_320(self, image):
        return resize_image_320(image)

    def resize_map(self, map, scale):
        return cv2.resize(map, None, fx=scale, fy=scale)

    def preprocess_image(self, image):
        return preprocess_image(image)

    def preprocess_group_entry(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)

        if self.option != 'reg_baseline_c5_dubreshko':
            # resize image
            image, image_scale = self.resize_image(image)
            # apply resizing to annotations too
            annotations[:, :2] *= image_scale
        else:
            # resize image
            image, image_scale_x, image_scale_y = self.resize_image_320(image)
            # apply resizing to annotations too
            annotations[:, 0] *= image_scale_x
            annotations[:, 1] *= image_scale_y

        # randomly transform image and annotations
        image_transeformed, annotation_transeformed = self.random_transform_rbg_centers_images(image, annotations)

        annotation_map = self.compute_keypoints_targets(image_transeformed.shape, annotation_transeformed)

        return image_transeformed, annotation_map


    def preprocess_group(self, image_group, annotations_group_leaves_center, annotations_group_num_of_leaves):
        annotations_group_leaves_center_a = annotations_group_leaves_center[0]

        for index, (image, annotations) in enumerate(zip(image_group, annotations_group_leaves_center_a)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)
            # copy processed data back to group
            image_group[index]       = image
            annotations_group_leaves_center[0][index] = annotations

        return image_group, annotations_group_leaves_center, annotations_group_num_of_leaves

    def preprocess_group_multi_maps(self, image_group, annotations_group_leaves_center, annotations_group_num_of_leaves):

        annotations_group_leaves_center_a = annotations_group_leaves_center[0]

        for index, (image, annotations) in enumerate(zip(image_group, annotations_group_leaves_center_a)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry_multi_maps(image, annotations)
            # copy processed data back to group
            image_group[index]       = image
            annotations_group_leaves_center[0][index] = annotations

        return image_group, annotations_group_leaves_center, annotations_group_num_of_leaves

    def show_annotations_on_image(self, image, annotations):
        import PIL.ImageDraw as ImageDraw
        import PIL.Image as Image
        import copy
        im_drawing = Image.fromarray(np.uint8(copy.deepcopy(image)))
        draw_mask = ImageDraw.Draw(im_drawing)
        for point in annotations:
            x_tl = round(int(point[0]) - 10)
            x_br = round(int(point[0]) + 10)
            y_tl = round(int(point[1]) - 10)
            y_br = round(int(point[1]) + 10)
            draw_mask.ellipse([x_tl, y_tl, x_br, y_br], fill='blue')
        im_drawing.show()

    def preprocess_group_entry_multi_maps(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)

        if self.option != 'reg_baseline_c5_dubreshko':
            # resize image
            image, image_scale = self.resize_image(image)
            # apply resizing to annotations too
            annotations[:, :2] *= image_scale
        else:
            # resize image
            image, image_scale_x, image_scale_y = self.resize_image_320(image)
            # apply resizing to annotations too
            annotations[:, 0] *= image_scale_x
            annotations[:, 1] *= image_scale_y

        # self.show_annotations_on_image(image, annotations)

        # randomly transform image and annotations
        image_transeformed, annotation_transeformed = self.random_transform_rbg_centers_images(image, annotations)

        # self.show_annotations_on_image(image_transeformed, annotation_transeformed)

        annotation_map_1 = self.compute_keypoints_targets_multi_maps(image_transeformed.shape, annotation_transeformed, radius = (3,3))
        annotation_map_2 = self.compute_keypoints_targets_multi_maps(image_transeformed.shape, annotation_transeformed, radius = (7,7))
        annotation_map_3 = self.compute_keypoints_targets_multi_maps(image_transeformed.shape, annotation_transeformed, radius = (5,5))
        annotation_map_4 = self.compute_keypoints_targets_multi_maps(image_transeformed.shape, annotation_transeformed, radius = (5,5))
        annotation_map_5 = self.compute_keypoints_targets_multi_maps(image_transeformed.shape, annotation_transeformed, radius = (3,3))

        annotation_map = [annotation_map_1,annotation_map_2,annotation_map_3,annotation_map_4,annotation_map_5]

        return image_transeformed, annotation_map


    def compute_keypoints_targets_multi_maps(self, image_shape, annotations_leaves_centers_a,radius=(5,5)):
        # resize transformed-image and annotations
        import copy
        annotations_leaves_centers = copy.deepcopy(annotations_leaves_centers_a)
        # here we should resize image too and then check it with the annotations
        output_shape = image_output_shape(image_shape)
        image_ratio = images_ratios(image_shape, output_shape)
        annotations_leaves_centers[:, :2] = annotations_leaves_centers[:, :2] * image_ratio
        annotations = np.zeros(output_shape)
        for i in range(annotations_leaves_centers.shape[0]):

            gaussian_map = create_gausian_mask(annotations_leaves_centers[i, :2], output_shape[1],output_shape[0], radius=radius)
            # each center point in the GT will be 1 in the annotation map
            annotations = np.maximum(annotations, gaussian_map)

        if np.isnan(annotations).any():
            raise("nan was found")
        return annotations

    def preprocess_group_input(self, image_group):
        for index, image in enumerate(image_group):
            # preprocess a single group entry
            # preprocess the image
            image = self.preprocess_image(image)
            # resize image
            image,_ = self.resize_image(image)

            # copy processed data back to group
            image_group[index]       = image

        return image_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    # TODO - modify to the relevant outputs types (points prediction and leaf counts) as the new targets
    def compute_targets(self, image_group, annotations_group_leaves_center, annotations_group_num_of_leaves):

        # compute labels and regression targets
        labels_group     = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations_leaves_centers, annotations_num_of_leaves) in enumerate(zip(image_group, annotations_group_leaves_center, annotations_group_num_of_leaves)):
            # compute regression targets
            labels_group[index] = annotations_leaves_centers[0]

            regression_group[index] = annotations_num_of_leaves[0][0]

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression

        if self.option == 'reg_baseline_c5' or self.option == 'reg_fpn_p3' or self.option == 'reg_baseline_c5_dubreshko':
            ret = [regression_batch]
        elif self.option == 'reg_fpn_p3_p7_avg' or self.option == 'reg_fpn_p3_p7_mle' or self.option == 'reg_fpn_p3_p7_min_sig' or self.option== 'reg_fpn_p3_p7_min_sig_L1' or self.option=='reg_fpn_p3_p7_mle_L1':
            ret = [regression_batch, regression_batch, regression_batch, regression_batch, regression_batch]
        return ret

    def compute_targets_multi_maps(self, image_group, annotations_group_leaves_center, annotations_group_num_of_leaves):

        # compute labels and regression targets
        labels_group = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations_leaves_centers, annotations_num_of_leaves) in enumerate(
                zip(image_group, annotations_group_leaves_center, annotations_group_num_of_leaves)):
            # compute regression targets
            labels_group[index] = annotations_leaves_centers[0]
            # self.compute_keypoints_targets(
            #     max_shape,
            #     annotations_leaves_centers,
            # )
            regression_group[index] = annotations_num_of_leaves[0][0]

        labels_batch_1 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_2 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_3 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_4 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())
        labels_batch_5 = np.zeros((self.batch_size,) + labels_group[0][0].shape, dtype=keras.backend.floatx())

        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch_1[index, ...] = labels[0]
            labels_batch_2[index, ...] = labels[1]
            labels_batch_3[index, ...] = labels[2]
            labels_batch_4[index, ...] = labels[3]
            labels_batch_5[index, ...] = labels[4]
            regression_batch[index, ...] = regression


        ret = [regression_batch, np.expand_dims(labels_batch_1, 3),
                                 np.expand_dims(labels_batch_2, 3),
                                 np.expand_dims(labels_batch_3, 3),
                                 np.expand_dims(labels_batch_4, 3),
                                 np.expand_dims(labels_batch_5, 3)]
        return ret


    def extract_centers_from_image(self, centers_image):
        annotations = []

        ######################################################################

        if "A4" not in self.csv_leaf_location_file:
            Ys, Xs, _ = np.nonzero(centers_image)
        else:
            Xs, Ys, _ = np.nonzero(centers_image)
        ######################################################################

        for index in range(len(Xs)):
            x = Xs[index]
            y = Ys[index]
            record = [x, y, 0.0]
            annotations.append(record)

        return annotations

    def compute_input_output(self, group):
        '''
        this function loads the image and preprocess it for a forward pass in the network and computes the needed output
        :param group: indices of the images for the current learning step. len(group) is determined by the batch size
        :return:
        '''

        # load images and annotations
        image_group       = self.load_image_group(group)
        #image_centers = self.load_centers_image_group(group)
        if self.csv_leaf_number_file:
            annotations_group_leaves_center = self.load_annotations_group_leaves_center(group)
            annotations_group_num_of_leaves = self.load_annotations_group_num_of_leaves(group)

            # check validity of annotations
            image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves = self.filter_annotations(image_group, annotations_group_leaves_center, annotations_group_num_of_leaves, group)

            # perform preprocessing steps
            if self.option == 20:
                # in this option each sub-network layer will have a different gaussian map as target
                image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves = self.preprocess_group_multi_maps(image_group, annotations_group_leaves_center, annotations_group_num_of_leaves)
            else:
                image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves = self.preprocess_group(
                    image_group, annotations_group_leaves_center, annotations_group_num_of_leaves)
        else:
            image_group_0 = self.preprocess_group_input(image_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group_0)

        if self.csv_leaf_number_file:
            # compute network targets
            if self.option == 20:
                targets = self.compute_targets_multi_maps(image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves)
            else:
                targets = self.compute_targets(image_group_0, annotations_group_leaves_center, annotations_group_num_of_leaves)
            return inputs, targets
        else:
            return inputs


    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)

