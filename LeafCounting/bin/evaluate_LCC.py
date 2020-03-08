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
import os
import sys

import keras
import tensorflow as tf
import random

from GetEnvVar import GetEnvVar


# Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# import keras_retinanet.bin
__package__ = "keras_retinanet.bin"


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models

from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.eval_LCC import evaluate
from ..utils.keras_version import check_keras_version
from ..preprocessing.csv_LCC_generator import CSVLCCGenerator


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    elif args.dataset_type == 'csv':
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
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    '''
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',             help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',   help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',   help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',  help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',       help='Path for saving images with detections.')
    parser.add_argument('--image-min-side',  help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',  help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    '''
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
    # print model summary
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
    # df = pd.read_csv(args.save_res_path)
    # df = df.append(pd.DataFrame({"Exp":[str(args.exp_num)],"Dataset":[args.dataset],"Dic":[str(CountDiff)], "AbsDic":[str(AbsCountDiff)], "Agreement":[str(CountAgreement)], "MSE":[str(MSE)]}),sort=False)
    # df.to_csv(args.save_res_path,index=False)
    if args.calc_det_performance:
        return CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap
    return CountDiff, AbsCountDiff, CountAgreement, MSE, R_2



if __name__ == '__main__':
    args = None

    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    args.dataset = 'BL'
    args.random_transform = True
    args.gpu = '0'

    args.pipe = 'keyPfinder' #'keyPfinder' #'reg'  #
    args.exp_num = 724
    eval_on_set = 'Test'
    if args.pipe == 'reg':
        '''
        reg options:
        'reg_baseline_c5_dubreshko'
        'reg_baseline_c5'
        'reg_fpn_p3'
        'reg_fpn_p3_p7_avg'
        'reg_fpn_p3_p7_mle'   this is our best regressor
        'reg_fpn_p3_p7_min_sig'
        '''
        args.option = 'reg_fpn_p3_p7_mle'
        args.calc_det_performance = False
    elif args.pipe == 'keyPfinder':
        args.option = 20
        args.calc_det_performance = False
    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        sys.exit

    args.dataset_type = 'csv'
    random.seed(10)

    if args.dataset == 'BL':
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets', 'Banana_leaves', args.dataset)

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'BL_exp_res', args.pipe, "results",
                                      'exp_' + str(args.exp_num))

        args.val_csv_leaf_number_file = os.path.join(args.data_path, 'test', args.dataset + '_Test.csv')
        args.val_csv_leaf_location_file = os.path.join(args.data_path, 'test', args.dataset + '_Test_leaf_location.csv')
    else:
        args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                      'CVPPP2017_LCC_training', 'training', args.dataset)

        args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                      'exp_' + str(args.exp_num) + 'testing')

        args.val_csv_leaf_number_file = os.path.join(args.data_path, args.dataset + '_Test', args.dataset + '_' + eval_on_set + '.csv')
        args.val_csv_leaf_location_file = os.path.join(args.data_path, args.dataset + '_Test', args.dataset + '_' + eval_on_set + '_leaf_location.csv')

    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None

    args.backbone = 'resnet50'
    args.batch_size = 1

    args.multi_gpu = False
    args.multi_gpu_force = False

    args.freeze_backbone = False

    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    # choose a cv number (1,...,4)
    cv_num = 1
    args.model = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                              'exp_' + str(args.exp_num), 'cv_' + str(cv_num), 'resnet50_csv.h5')

    CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = main(args)

    # for cv_num in range(1,5):
    #
    #     if eval_on_set=='Val':
    #         args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', args.pipe, "results",
    #                                       'exp_' + str(args.exp_num))  # , 'cv_'+str(cv_num)
    #
    #         args.val_csv_leaf_number_file = os.path.join(args.data_path, args.dataset + '_cv'+'_exp_' + str(
    #             args.exp_num) + '_' + eval_on_set + '.csv')
    #         args.val_csv_leaf_location_file = os.path.join(args.data_path, args.dataset +  '_cv'+ '_exp_' + str(
    #                                                            args.exp_num) + '_' + eval_on_set + '_leaf_location.csv')
    #
    #         args.model = os.path.join(GetEnvVar('ModelsPath'), 'LCC_Models_senepshots', args.pipe,
    #                                   'exp_' + str(args.exp_num), 'resnet50_csv.h5')
    #     else:
    #         args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', args.pipe, "results",
    #                                       'exp_' + str(args.exp_num), 'cv_'+str(cv_num))
    #
    #         args.val_csv_leaf_number_file = os.path.join(args.data_path, args.dataset + '_cv'+ str(cv_num)+'_exp_' + str(
    #             args.exp_num)+ '_' + eval_on_set+ '.csv')  # os.path.join(args.data_path, 'Ac.csv') #
    #         args.val_csv_leaf_location_file = os.path.join(args.data_path, args.dataset + '_cv' +str(cv_num)+ '_exp_' + str(
    #             args.exp_num)+ '_'+eval_on_set+'_leaf_location.csv')  # os.path.join(args.data_path, 'Ac_leaf_location.csv') #
    #
    #         args.model = os.path.join(GetEnvVar('ModelsPath'), 'LCC_Models_senepshots', args.pipe,
    #                                   'exp_' + str(args.exp_num), 'cv_'+str(cv_num) ,'resnet50_csv.h5')
    #
    #     if args.calc_det_performance:
    #         CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = main(args)
    #     else:
    #         CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = main(args)

