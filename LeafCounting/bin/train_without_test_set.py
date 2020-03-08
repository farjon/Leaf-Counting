import csv
import random
import sys
import os
import argparse
import numpy as np
from GetEnvVar import GetEnvVar
import train
import train_reg
import evaluate_LCC
import create_csv_of_leaf_center
import pandas as pd


#Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import keras_retinanet.bin
__package__ = "keras_retinanet.bin"



def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    return parser.parse_args(args)


def get_paths_dict(Ac_files_path, exp_id):

    train_count_file = os.path.join(Ac_files_path, 'Ac_cv' + '_exp_' + str(exp_id) + '_Train.csv')
    train_centers_file = os.path.join(Ac_files_path, 'Ac_cv' + '_exp_' + str(exp_id) + '_Train_leaf_location.csv')

    val_count_file = os.path.join(Ac_files_path, 'Ac_cv' + '_exp_' + str(exp_id) + '_Val.csv')
    val_centers_file = os.path.join(Ac_files_path, 'Ac_cv' + '_exp_' + str(exp_id) + '_Val_leaf_location.csv')

    files_paths = {}
    files_paths['train_count_file'] = train_count_file
    files_paths['train_centers_file'] = train_centers_file

    files_paths['val_count_file'] = val_count_file
    files_paths['val_centers_file'] = val_centers_file

    return files_paths


def data_split_for_Ac(DATASET_DIR, dataset):
    leaf_counts = []

    # read the leaf counts csv file
    csvPath = os.path.join(DATASET_DIR, dataset + ".csv")
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on spliting dataset: ", dataset, "\n")
        count = 0
        for row in readCSV:
            print(row)
            rgbImage_name = row[0]
            plant_name = dataset+'_'+ rgbImage_name

            current_leaf_count = {}
            current_leaf_count[plant_name] = int(row[1])
            current_leaf_count = [current_leaf_count]
            leaf_counts.append(current_leaf_count)
            count += 1
        print()

    print("Done, ", dataset, "set - has", count, "images \n")


    leaf_location_csvPath = os.path.join(DATASET_DIR, dataset + "_leaf_location.csv")

    # create the centers coordinates csv, if doesn't exist yet - of the whole dataset
    if os.path.isfile(leaf_location_csvPath) == False:
        create_csv_of_leaf_center.main(DATASET_DIR, dataset)

    # create a list where each item is a pair: plant key, list of center coordinates
    # the center coordinates of the leaves of each image at each dataset
    coord_dict = get_centers_data_for_Ac(dataset, leaf_location_csvPath)

    leaf_location_coord = []
    for key, value in coord_dict.items():
        leaf_location_coord.append([key, value])


    print("Done reading the datasets, start random split of the data... \n")

    # Create a random datasets split
    num_of_images = len(leaf_location_coord)
    N_train = round(0.8* num_of_images)
    N_val = num_of_images-N_train

    np.random.seed(0)

    Perm = np.random.permutation(num_of_images)  # Randomly permute a sequence
    train_inx = Perm[0:N_train]  # indices for train
    val_inx = Perm[N_train:]     # indices for fold 4

    # sorting the lists so they will be correlated
    leaf_counts[0].sort()
    leaf_location_coord.sort()

    # Create train data
    train_leaf_counts = [leaf_counts[i] for i in train_inx]
    train_leaf_location_coord = [leaf_location_coord[i] for i in train_inx]

    # Create val data
    val_leaf_counts = [leaf_counts[i] for i in val_inx]
    val_leaf_location_coord = [leaf_location_coord[i] for i in val_inx]


    print("Done splitting the data..")
    print("Total num of images: ", num_of_images)
    print("Num of train images: ", len(train_inx))
    print("Num of val images: ", len(val_inx))

    print()

    All_Splitted_Data = {}

    All_Splitted_Data["train_leaf_counts"] = train_leaf_counts
    All_Splitted_Data["train_leaf_location_coord"] = train_leaf_location_coord

    All_Splitted_Data["val_leaf_counts"] = val_leaf_counts
    All_Splitted_Data["val_leaf_location_coord"] = val_leaf_location_coord

    return All_Splitted_Data



def get_current_data_for_sub(All_Splitted_Data):

    Train_leaf_counts = All_Splitted_Data['train_leaf_counts']
    Train_leaf_location_coord = All_Splitted_Data['train_leaf_location_coord']

    Val_leaf_counts = All_Splitted_Data['val_leaf_counts']
    Val_leaf_location_coord = All_Splitted_Data['val_leaf_location_coord']


    current_data_dict = {}
    current_data_dict['Train_leaf_counts'] = Train_leaf_counts
    current_data_dict['Train_leaf_location_coord'] = Train_leaf_location_coord
    current_data_dict['Val_leaf_counts'] = Val_leaf_counts
    current_data_dict['Val_leaf_location_coord'] = Val_leaf_location_coord


    return current_data_dict



def get_data(files_paths, all_data_path):

    # Ac files for current fold

    train_count_data = []
    train_centers_data = []

    val_count_data = []
    val_centers_data = []


    all_Ac_files = [files_paths['train_count_file'], files_paths['train_centers_file'], files_paths['val_count_file'], files_paths['val_centers_file']]

    # delete previous files if exist
    for file_path in all_Ac_files:
        if os.path.isfile(file_path):
            os.remove(file_path)


    data_sets = ['A1', 'A2', 'A3', 'A4']

    for ds in data_sets:

        args.data_path = os.path.join(all_data_path, ds)

        All_Splitted_Data = data_split_for_Ac(args.data_path, ds)


        current_data_dict = get_current_data_for_sub(All_Splitted_Data)

        for value in current_data_dict['Train_leaf_counts']:
            train_count_data.append(value)
        for value in current_data_dict['Train_leaf_location_coord']:
            train_centers_data.append(value)

        for value in current_data_dict['Val_leaf_counts']:
            val_count_data.append(value)
        for value in current_data_dict['Val_leaf_location_coord']:
            val_centers_data.append(value)


    create_sub_csv_file_for_Ac(files_paths['train_count_file'], files_paths['train_centers_file'], train_count_data, train_centers_data)
    create_sub_csv_file_for_Ac(files_paths['val_count_file'], files_paths['val_centers_file'], val_count_data, val_centers_data)



def get_centers_data_for_Ac(data, leaf_location_csvPath):

    coord_dict = {}

    # read the leaf_location csv file
    with open(leaf_location_csvPath) as csvfile_2:
        readCSV_2 = csv.reader(csvfile_2)
        print("Reading leaf coordinates: ", "\n")
        # create a dictionary for the center coordinates of each plant in each dataset
        for row in readCSV_2:
            plant_name = row[0]
            x = int(row[1])
            y = int(row[2])
            #key = data + "_" + plant_name
            key = data+'_'+plant_name
            if len(coord_dict) == 0:
                coord_dict[key] = []
                coord_dict[key].append([x, y])
            else:
                if key in coord_dict.keys():
                    coord_dict[key].append([x, y])
                else:
                    coord_dict[key] = []
                    coord_dict[key].append([x, y])

    return coord_dict



def create_sub_csv_file_for_Ac(csv_leaf_number_file, csv_leaf_location_file, leaf_counts, leaf_location_coord ):
    '''
    input: indices of images
    output: generates csv file of a subset from the given file, based on the required indices
    '''
    # Create a csv file of leaf counts for the relevant set

    new_counts_file_path = csv_leaf_number_file
    with open(new_counts_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(leaf_counts)):
            line = leaf_counts[i][0]
            keys = line.keys()
            for key in keys:
                count = line[key]
                writer.writerow([key, count])

    # Create a csv file of center points for the relevant set
    new_centers_file_path = csv_leaf_location_file
    with open(new_centers_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(leaf_location_coord)):
            line = leaf_location_coord[i]
            name = line[0]
            points = line[1]
            for j in range(len(points)):
                x = points[j][0]
                y = points[j][1]
                writer.writerow([name, x, y])




def main(args=None):

    random.seed(50)

    args.random_transform = True

    args.pipe = 'reg' #'keyPfinder' #'reg'

    args.exp_num = 141013

    args.early_stopping_indicator = "AbsCountDiff"
    args.epochs = 2
    args.gpu = '0'

    args.exp_name = 'hyper_1'

    args.early_stopping_patience = 1

    # important?
    args.multi_gpu = False
    args.multi_gpu_force = False

    if args.pipe == 'reg':
        args.option = 0
        args.calc_det_performance = False
    elif args.pipe == 'keyPfinder':
        args.option = 10
        args.calc_det_performance = True
    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        return


    all_data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Phenotyping Datasets', 'Plant phenotyping', 'data_2',
                                 'CVPPP2017_LCC_training', 'training')

    Ac_files_path = os.path.join(all_data_path, 'Ac')

    args.snapshot_path = os.path.join(GetEnvVar('ModelsPath'), 'LCC_Models_senepshots', args.pipe,
                                      'exp_' + str(args.exp_num))
    args.tensorboard_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', args.pipe, 'log_dir',
                                        'exp_' + str(args.exp_num))

    args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'LCC_exp_res', args.pipe, "results",
                                  'exp_' + str(args.exp_num))

    files_paths = get_paths_dict(Ac_files_path, args.exp_num)

    get_data(files_paths, all_data_path)


    train_count_file = files_paths['train_count_file']
    train_centers_file = files_paths['train_centers_file']

    val_count_file = files_paths['val_count_file']
    val_centers_file = files_paths['val_centers_file']

    args.train_csv_leaf_number_file = train_count_file
    args.train_csv_leaf_location_file = train_centers_file

    args.val_csv_leaf_number_file = val_count_file
    args.val_csv_leaf_location_file = val_centers_file

    #Train the model based on current split
    print('Start training on Ac')
    if args.pipe == 'keyPfinder':
        train.main(args)

    elif args.pipe == 'reg':
        train_reg.main(args)


    print("Done")



if __name__ == '__main__':
    args = None
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    args.snapshot = None
    args.imagenet_weights = True
    args.weights = None

    args.backbone = 'resnet50'
    args.batch_size = 1

    args.freeze_backbone = False

    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    args.dataset_type = 'csv'

    main(args)
