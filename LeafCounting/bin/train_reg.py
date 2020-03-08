#!/usr/bin/env python

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

import argparse
import functools
import os
import sys
import warnings

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf

from GetEnvVar import GetEnvVar

# # Allow relative imports when being executed as script.
# if __name__ == "__main__" and __package__ is None:
#     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#     import keras_retinanet.bin
#     __package__ = "keras_retinanet.bin"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#import keras_retinanet.bin
__package__ = "LeafCounting.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.LLC_eval import Evaluate_LLCtype
from ..models.gyf_net_reg import gyf_net_LCC
from ..preprocessing.csv_LCC_generator import CSVLCCGenerator
from ..utils.anchors import make_shapes_callback, anchor_targets_bbox
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_gyf_net, num_classes, weights, option, current_lr, do_dropout, multi_gpu=0, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_gyf_net(num_classes, option=option, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_gyf_net(num_classes, option=option, modifier=modifier, do_dropout= do_dropout), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = gyf_net_LCC(model=model, option=option)

    if option == 'reg_baseline_c5' or option == 'reg_fpn_p3' or option == 'reg_baseline_c5_dubreshko':
        vriable_losses = {'regression': keras.losses.mse}
    if option == 'reg_fpn_p3_p7_avg':
        vriable_losses = {'FC_submodel': keras.losses.mse}
    if option == 'reg_fpn_p3_p7_mle' or option == 'reg_fpn_p3_p7_min_sig':
        vriable_losses = {'FC_submodel' : losses.mu_sig_gyf()}
    if option == 'reg_fpn_p3_p7_mle_L1' or option == 'reg_fpn_p3_p7_min_sig_L1':
        vriable_losses = {'FC_submodel' : losses.mu_sig_gyf_L1()}

    # compile model
    training_model.compile(
        loss=vriable_losses,
        optimizer=keras.optimizers.adam(lr=current_lr, clipnorm=0.001)
    )
    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        evaluation = Evaluate_LLCtype(validation_generator, tensorboard=tensorboard_callback, save_path = args.save_path)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    if args.option != 'reg_baseline_c5_dubreshko':
        # save the model
        if args.snapshot_path:
            # ensure directory created first; otherwise h5py will error after epoch.
            makedirs(args.snapshot_path)
            checkpoint = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    args.snapshot_path,
                    '{backbone}_csv.h5'.format(backbone=args.backbone)
                ),
                verbose=1,
                period= 1,
                save_best_only=True,
                monitor="mse",
                mode='min'
            )
            checkpoint = RedirectModel(checkpoint, model)
            callbacks.append(checkpoint)

        if args.early_stopping_indicator:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='mse',
                min_delta=0,
                patience=args.early_stopping_patience,
                verbose=0,
                mode='min'
            )
            early_stopping = RedirectModel(early_stopping, model)
            callbacks.append(early_stopping)

        if args.reduce_lr:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=args.reduceLR_factor,  # 0.1,
                patience=args.reduceLR_patience,  # 3
                verbose=1,
                mode='auto',
                epsilon=0.0001,
                cooldown=0,
                min_lr=0
            ))

    else:
        # save the model
        if args.snapshot_path:
            # ensure directory created first; otherwise h5py will error after epoch.
            makedirs(args.snapshot_path)
            checkpoint = keras.callbacks.ModelCheckpoint(
                os.path.join(
                    args.snapshot_path,
                    '{backbone}_csv.h5'.format(backbone=args.backbone)
                ),
                verbose=1,
                period=1,
                save_best_only=True,
                monitor="AbsCountDiff",
                mode='min'
            )
            checkpoint = RedirectModel(checkpoint, model)
            callbacks.append(checkpoint)

        if args.early_stopping_indicator:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor=args.early_stopping_indicator,
                min_delta=0,
                patience=args.early_stopping_patience,  # 50
                verbose=0,
                mode='min'
            )
            early_stopping = RedirectModel(early_stopping, model)
            callbacks.append(early_stopping)

    callbacks.append(
        keras.callbacks.TerminateOnNaN()
    )

    return callbacks


def create_generators(args):
    if args.option != 'reg_baseline_c5_dubreshko':
        # create random transform generator for augmenting training data
        if args.random_transform:
            transform_generator = random_transform_generator(
                min_rotation=-0.2,
                max_rotation=0.2,
                #min_translation=(-0.2, -0.2),
                #max_translation=(0.2, 0.2),
                #min_shear=-0.1,
                #max_shear=0.1,
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
        else:
            transform_generator = None

        train_generator = CSVLCCGenerator(
            args.train_csv_leaf_number_file,
            args.train_csv_leaf_location_file,
            args.option,
            # base_dir=args.data_path,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )

        if args.val_csv_leaf_number_file:
            validation_generator = CSVLCCGenerator(
                args.val_csv_leaf_number_file,
                args.val_csv_leaf_location_file,
                args.option,
                # base_dir=args.data_path,
                batch_size=args.batch_size,
                image_min_side=args.image_min_side,
                image_max_side=args.image_max_side
            )
        else:
            validation_generator = None
    else:
        transform_generator = random_transform_generator(
            min_rotation=-0.5,
            max_rotation=0.5,
            #min_scaling=(0.9, 0.9),
            max_scaling=(1.2, 1.2),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )

        train_generator = CSVLCCGenerator(
            args.train_csv_leaf_number_file,
            args.train_csv_leaf_location_file,
            args.option,
            # base_dir=args.data_path,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=320,
            image_max_side=320
        )

        if args.val_csv_leaf_number_file:
            validation_generator = CSVLCCGenerator(
                args.val_csv_leaf_number_file,
                args.val_csv_leaf_location_file,
                args.option,
                # base_dir=args.data_path,
                batch_size=args.batch_size,
                image_min_side=320,
                image_max_side=320
            )
        else:
            validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to retina_backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    if parsed_args.tensorboard_dir:
        makedirs(parsed_args.tensorboard_dir)

#    makedirs(parsed_args.data_path)

    return parsed_args


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    return parser.parse_args(args)

def csv_list(string):
    return string.split(',')


def main(args=None):

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # check that all the args exists
    args = check_args(args)

    # create object that stores backbone information

    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)
    # if args.option != 'reg_baseline_c5_dubreshko':
    #     args.steps = 5*int(train_generator.size() // args.batch_size)
    # else:
    #     args.steps = int(train_generator.size() // args.batch_size)

    args.steps = args.step_multi*int(train_generator.size() // args.batch_size)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        prediction_model = gyf_net_LCC(model=model, option=args.option)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_gyf_net=backbone.gyf_net_reg,
            num_classes=1,
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            option=args.option,
            current_lr = args.lr,
            do_dropout = args.do_dropout
        )

    # print model summary
    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        if validation_generator is not None:
            validation_generator.compute_anchor_targets = compute_anchor_targets

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    # store final result too
    model.save(os.path.join(args.snapshot_path, 'resnet50_final.h5'))

if __name__ == '__main__':
    args = None

    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    args.pipe = 'reg'
    # path to the data (that organized as the data in the below folder)
    args.dataset = 'BL'
    args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Phenotyping Datasets', 'Plant phenotyping', 'data_2',
                                  'CVPPP2017_LCC_training', 'training', '{}'.format(args.dataset))

    if args.dataset == 'BL':
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets', 'Banana_leaves', args.dataset)

        args.train_csv_leaf_number_file = os.path.join(args.data_path, 'train', args.dataset+'_Train.csv')
        args.train_csv_leaf_location_file = os.path.join(args.data_path, 'train', args.dataset+'_Train_leaf_location.csv')
        args.val_csv_leaf_number_file = os.path.join(args.data_path, 'val', args.dataset+'_Val.csv')
        args.val_csv_leaf_location_file = os.path.join(args.data_path, 'val', args.dataset+'_Val_leaf_location.csv')

    else:
        ds = args.dataset
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets', 'CVPPP2017_LCC_training', 'training', ds)

        args.train_csv_leaf_number_file = os.path.join(args.data_path, 'train', ds+'_Train.csv')
        args.train_csv_leaf_location_file = os.path.join(args.data_path, 'train', ds+'_Train_leaf_location.csv')
        args.val_csv_leaf_number_file = os.path.join(args.data_path, 'val', ds+'_Val.csv')
        args.val_csv_leaf_location_file = os.path.join(args.data_path, 'val', ds+'_Val_leaf_location.csv')



    # To choose one of the below 3 options
    # To start with a pre-trained model - add link to the .h5 snapshot location or None
    args.snapshot = None
    # To start with a pre-trained (on Imagenet) resnet model - True \ False
    args.imagenet_weights = True
    # To start with a pre-trained model - add link to the .h5 weights file location or None
    args.weights = None

    args.exp_num = 525111
    # the model options are:
    '''
    reg options:
    'reg_baseline_c5_dubreshko'
    'reg_baseline_c5'
    'reg_fpn_p3'
    'reg_fpn_p3_p7_avg'
    'reg_fpn_p3_p7_mle'
    'reg_fpn_p3_p7_min_sig'
    'reg_fpn_p3_p7_mle_L1'
    'reg_fpn_p3_p7_min_sig_L1'
    '''
    args.option = 'reg_fpn_p3_p7_mle'
    # Backbone options are resnet50 / resnet101 / resnet152 (checked - resnet50)
    args.backbone = 'resnet50'
    args.batch_size = 1
    # Choose the NVIDIA-SMI gpu number
    args.gpu = '0'
    args.multi_gpu = 0 #(checked - only 0) the number of gpus to use
    args.multi_gpu_force = False #(checked - only False) to change to true if use more than 1 gpu

    # Maximum train epochs
    args.epochs = 100
    # the number of epochs between each validation check
    args.step_multi = 1
    # to freeze backbone (resnet) weights during train or not
    args.freeze_backbone = False
    # to use transformations (as describe in the generator function) during train or not
    args.random_transform = True
    # To do evaluation after eche epoch (True \ False)
    args.evaluation = True
    # The indicator for early stopping can be: AbsCountDiff (mae) or CountDiff (mean error) or mse
    args.early_stopping_indicator = "AbsCountDiff"
    # the number of epochs that the early_stopping_indicator dont improve but the train dont stop
    args.early_stopping_patience = 50
    # initial values of hyper-params - unless changed via hyper parameters tuning
    args.lr = 1e-5
    args.reduce_lr = True
    args.reduceLR_patience = 8
    args.reduceLR_factor = 0.05
    # choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333
    args.do_dropout = False
    args.snapshot_path = os.path.join(GetEnvVar('ModelsPath'), 'LCC_Models_senepshots', 'reg', 'exp_' + str(args.exp_num))
    args.tensorboard_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', 'reg', 'log_dir',
                                        'exp_' + str(args.exp_num))
    args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', 'reg', "results", 'exp_' + str(args.exp_num))
    args.save_res_path = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', args.pipe, "results",
                                      "results_temp.csv")


    main(args)
