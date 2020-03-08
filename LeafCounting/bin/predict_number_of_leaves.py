#!/usr/bin/env python

import argparse
import os
import sys
import keras
import tensorflow as tf
import random
from GetEnvVar import GetEnvVar


# Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
__package__ = "keras_retinanet.bin"


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..utils.eval_LCC import _get_prediction
from ..utils.eval_LCC import evaluate
from ..utils.keras_version import check_keras_version
from ..preprocessing.csv_LCC_generator import CSVLCCGenerator



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):

    validation_generator = CSVLCCGenerator(
        args.val_csv_leaf_number_file,
        args.val_csv_leaf_location_file,
        args.option,
        base_dir=args.data_path,
        batch_size=args.batch_size,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side
    )

    return validation_generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    return parser.parse_args(args)


def main(args=None):

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    print(model.summary())

    # make prediction model
    if args.pipe == 'reg':
        from ..models.gyf_net_reg import gyf_net_LCC
    elif args.pipe == 'keyPfinder':
        from ..models.gyf_net_keyPfinder import gyf_net_LCC
    else:
        print('Wrong pipe name - should be reg or keyPfinder')
        return

    model = gyf_net_LCC(model=model, option=args.option)

    # start evaluation

    if args.calc_det_performance:
        CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = evaluate(
            args.option,
            args.val_csv_leaf_number_file,
            generator,
            model,
            save_path=args.save_path,
            calc_det_performance=args.calc_det_performance
        )
    else:
        CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = evaluate(
            args.option,
            args.val_csv_leaf_number_file,
            generator,
            model,
            save_path=args.save_path,
            calc_det_performance=args.calc_det_performance
        )
    print("CountDiff:", CountDiff, "AbsCountDiff" ,AbsCountDiff, "CountAgreement", CountAgreement, "MSE", MSE)

    if args.calc_det_performance:
        return CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap
    return CountDiff, AbsCountDiff, CountAgreement, MSE, R_2




if __name__ == '__main__':

    args = sys.argv[1:]
    args = parse_args(args)
    args.random_transform = True

    args.dataset_type = 'csv'
    random.seed(10)

    # reg options:
        # reg_baseline_c5_dubreshko
        # reg_baseline_c5
        # reg_fpn_p3
        # reg_fpn_p3_p7_avg
        # reg_fpn_p3_p7_mle
        # reg_fpn_p3_p7_min_sig
        # reg_fpn_p3_p7_mle_L1
        # reg_fpn_p3_p7_min_sig_L1
    # keyPfinder options:
        # detection_option_20

    args.pipe = 'keyPfinder' #'keyPfinder' #'reg'
    args.exp_num = 724
    eval_on_set = 'Test'
    if args.pipe == 'reg':
        args.option = 'reg_fpn_p3_p7_mle'
        args.calc_det_performance = False
    elif args.pipe == 'keyPfinder':
        args.option = 20
        args.calc_det_performance = False
    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        sys.exit

    args.dataset = 'BL'

    if args.dataset == 'BL':
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets', 'Banana_leaves', args.dataset)

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'BL_exp_res', args.pipe, "results",
                                      'exp_' + str(args.exp_num))

    else:
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                      'CVPPP2017_LCC_training', 'training', args.dataset)

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                      'exp_' + str(args.exp_num) + 'testing')

    args.val_csv_leaf_number_file = None
    args.val_csv_leaf_location_file = None

    cv_num = 1
    args.model = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                              'exp_' + str(args.exp_num), 'cv_' + str(cv_num), 'resnet50_csv.h5')

    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None

    args.backbone = 'resnet50'
    args.batch_size = 1

    args.gpu = '0'
    args.multi_gpu = False
    args.multi_gpu_force = False

    args.freeze_backbone = False
    args.evaluation = True
    args.image_min_side = 800
    args.image_max_side = 1333

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    generator = create_generator(args)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)
    # print model summary
    print(model.summary())

    # make prediction model
    if args.pipe == 'reg':
        from ..models.gyf_net_reg import gyf_net_LCC
    elif args.pipe == 'keyPfinder':
        from ..models.gyf_net_keyPfinder import gyf_net_LCC


    model = gyf_net_LCC(model=model, option=args.option)

    # start evaluation

    predictions = _get_prediction(
        args.option,
        generator,
        model,
        save_path=args.save_path,
        calc_det_performance=args.calc_det_performance)

    # If you want - save predictions here



