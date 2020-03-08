import csv
import random
import sys
import os
import argparse
import numpy as np
from GetEnvVar import GetEnvVar
import pandas as pd

#Allow relative imports when being executed as script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
__package__ = "LeafCounting.bin"

import train
import train_reg
import evaluate_LCC


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    return parser.parse_args(args)


def get_centers_data(leaf_location_csvPath):

    coord_dict = {}

    # read the leaf_location csv file
    with open(leaf_location_csvPath) as csvfile_2:
        readCSV_2 = csv.reader(csvfile_2)
        # create a dictionary for the center coordinates of each plant in each dataset
        for row in readCSV_2:
            plant_name = row[0].split("_")[0]
            x = int(row[1])
            y = int(row[2])
            #key = data + "_" + plant_name
            key = plant_name
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

def read_leaf_count(dataset_path, dataset):
    leaf_count = []

    # read the leaf counts csv file
    csvPath = os.path.join(dataset_path, dataset + ".csv")
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile)
        print("Working on spliting dataset: ", dataset, "\n")
        count = 0
        for row in readCSV:
            print(row)
            rgbImage_name = row[0]
            if dataset == 'BL':
                plant_name = rgbImage_name.split(".")[0]
            else:
                plant_name = rgbImage_name.split("_")[0]

            current_leaf_count = {}
            current_leaf_count[plant_name] = int(row[1])
            current_leaf_count = [current_leaf_count]
            leaf_count.append(current_leaf_count)
            count += 1
        print()

    print("Done, ", dataset, "set - has", count, "images \n")
    return leaf_count

def read_lead_location(dataset_path, dataset_name):

    leaf_location_csvPath = os.path.join(dataset_path, dataset_name + "_leaf_location.csv")
    # create a list where each item is a pair: plant key, list of center coordinates
    # the center coordinates of the leaves of each image at each dataset
    coord_dict = get_centers_data(leaf_location_csvPath)

    leaf_location_coord = []
    for key, value in coord_dict.items():
        leaf_location_coord.append([key, value])

    return leaf_location_coord

def split_to_folds(leaf_count, leaf_location_coord):
    All_Splitted_Data = {}

    # Create a random datasets split
    num_of_images = len(leaf_location_coord)
    NOI_per_fold = round((1/4) * num_of_images) # 4 represents number of folds

    Perm = np.random.permutation(num_of_images)  # Randomly permute a sequence

    # sorting the lists so they will be correlated
    leaf_count[0].sort()
    leaf_location_coord.sort()

    # Create Fold1 data
    fold1_ind = Perm[0:NOI_per_fold]  # indices for fold 1
    All_Splitted_Data["f1_leaf_counts"] = [leaf_count[i] for i in fold1_ind]
    All_Splitted_Data["f1_leaf_location_coord"] = [leaf_location_coord[i] for i in fold1_ind]

    # Create Fold2 data
    fold2_ind = Perm[NOI_per_fold: NOI_per_fold * 2]  # indices for fold 2
    All_Splitted_Data["f2_leaf_counts"] = [leaf_count[i] for i in fold2_ind]
    All_Splitted_Data["f2_leaf_location_coord"] = [leaf_location_coord[i] for i in fold2_ind]

    # Create Fold3 data
    fold3_ind = Perm[NOI_per_fold * 2: NOI_per_fold * 3]  # indices for fold 3
    All_Splitted_Data["f3_leaf_counts"] = [leaf_count[i] for i in fold3_ind]
    All_Splitted_Data["f3_leaf_location_coord"] = [leaf_location_coord[i] for i in fold3_ind]

    # Create Fold4 data
    fold4_ind = Perm[NOI_per_fold * 3:]  # indices for fold 4
    All_Splitted_Data["f4_leaf_counts"] = [leaf_count[i] for i in fold4_ind]
    All_Splitted_Data["f4_leaf_location_coord"] = [leaf_location_coord[i] for i in fold4_ind]

    print("Done splitting the data..")

    return All_Splitted_Data


def data_split(dataset_path, dataset_name):
    # read annotations files
    leaf_count = read_leaf_count(dataset_path, dataset_name)
    leaf_location_coord = read_lead_location(dataset_path, dataset_name)

    print("Done reading the datasets, start random split of the data... \n")

    All_Splitted_Data = split_to_folds(leaf_count, leaf_location_coord)

    return All_Splitted_Data


def create_sub_csv_file(csv_leaf_number_file, csv_leaf_location_file, leaf_counts, leaf_location_coord, ds):
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
                if ds == 'BL':
                    name = key + '.jpg'
                else:
                    name = key + "_rgb.png"
                writer.writerow([name, count])

    # Create a csv file of center points for the relevant set
    new_centers_file_path = csv_leaf_location_file
    with open(new_centers_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(leaf_location_coord)):
            line = leaf_location_coord[i]
            if ds == 'BL':
                name = line[0]
            else:
                name = line[0] + "_centers.png"
            points = line[1]
            for j in range(len(points)):
                x = points[j][0]
                y = points[j][1]
                writer.writerow([name, x, y])



def split_to_sets(All_Splitted_Data, Test_fold_num, Val_fold_num, Train_fold_num):

    Train_leaf_counts_1 = All_Splitted_Data['f' + str(Train_fold_num[0]) + '_leaf_counts']
    Train_leaf_location_coord_1 = All_Splitted_Data['f' + str(Train_fold_num[0]) + '_leaf_location_coord']
    Train_leaf_counts_2 = All_Splitted_Data['f' + str(Train_fold_num[1]) + '_leaf_counts']
    Train_leaf_location_coord_2 = All_Splitted_Data['f' + str(Train_fold_num[1]) + '_leaf_location_coord']

    current_data_dict = {}
    current_data_dict['Train_leaf_counts'] = Train_leaf_counts_1 + Train_leaf_counts_2
    current_data_dict['Train_leaf_location_coord'] = Train_leaf_location_coord_1 + Train_leaf_location_coord_2

    current_data_dict['Val_leaf_counts'] = All_Splitted_Data['f' + str(Val_fold_num) + '_leaf_counts']
    current_data_dict['Val_leaf_location_coord'] = All_Splitted_Data['f' + str(Val_fold_num) + '_leaf_location_coord']

    current_data_dict['Test_leaf_counts'] = All_Splitted_Data['f' + str(Test_fold_num) + '_leaf_counts']
    current_data_dict['Test_leaf_location_coord'] = All_Splitted_Data['f' + str(Test_fold_num) + '_leaf_location_coord']

    return current_data_dict

def get_aggregated_results_withHyper(args, stats, ds):

    all_CountDiff = stats[ds]['CountDiff']
    all_AbsCountDiff = stats[ds]['AbsCountDiff']
    all_CountAgreement = stats[ds]['CountAgreement']
    all_MSE = stats[ds]['MSE']
    all_R_2 = stats[ds]['R_2']
    all_ap = stats[ds]['ap']

    mean_CountDiff = np.mean(all_CountDiff)
    mean_AbsCountDiff = np.mean(all_AbsCountDiff)
    mean_CountAgreement = np.mean(all_CountAgreement)
    mean_MSE = np.mean(all_MSE)
    mean_R_2 = np.mean(all_R_2)

    std_CountDiff = np.std(all_CountDiff)
    std_AbsCountDiff = np.std(all_AbsCountDiff)
    std_CountAgreement = np.std(all_CountAgreement)
    std_MSE = np.std(all_MSE)
    std_R_2 = np.std(all_R_2)

    #ap values
    if args.calc_det_performance:
        mean_ap = round(np.mean(all_ap), 3)
        std_ap = round(np.std(all_ap), 3)
    else:
        mean_ap = None
        std_ap = None

    print()
    print('All CV results:', ds)
    print('The CountDiff values:', [ '%.3f' % elem for elem in all_CountDiff ])
    print('The AbsCountDiff values:', [ '%.3f' % elem for elem in all_AbsCountDiff])
    print('The CountAgreement values:', [ '%.3f' % elem for elem in all_CountAgreement])
    print('The MSE values:', [ '%.3f' % elem for elem in all_MSE])
    print('The R_2 values:', ['%.3f' % elem for elem in all_R_2])

    print('early_stopping_patience:', args.early_stopping_patience)
    print('reduceLR_patience:', args.reduceLR_patience)
    print('reduceLR_factor:', args.reduceLR_factor)
    print('lr:', args.lr)

    if args.calc_det_performance:
        print('The ap values:', ['%.3f' % elem for elem in all_ap])
    else:
        print('The ap values:', 'None')

    print()

    print('Summarized results for: train on:', ds)
    print('mean_CountDiff:', round(mean_CountDiff, 3), 'std_CountDiff:', round(std_CountDiff, 3))
    print('mean_AbsCountDiff:', round(mean_AbsCountDiff,3), 'std_AbsCountDiff:', round(std_AbsCountDiff, 3))
    print('mean_CountAgreement:', round(mean_CountAgreement, 3), 'std_CountAgreement:', round(std_CountAgreement, 3))
    print('mean_MSE:', round(mean_MSE, 3), 'std_MSE:', round(std_MSE, 3))
    print('mean_R_2:', round(mean_R_2, 3), 'std_R_2:', round(std_R_2, 3))

    if args.calc_det_performance:
        print('mean_ap:', round(mean_ap, 3), 'std_ap:', round(std_ap, 3))
        ap_data =[['%.3f' % elem for elem in all_ap]]
    else:
        print('mean_ap:', 'None', 'std_ap:', 'None')
        ap_data = ['None']


    df = pd.read_csv(args.save_res_path)
    new_data = pd.DataFrame({"Exp":[str(args.exp_num)], "Augmantation": str(args.random_transform), "Train_set":[ds], "Test_set":[ds],
                             "mean_Dic":[str(round(mean_CountDiff, 3))], "std_Dic":[str(round(std_CountDiff, 3))],
                             "mean_AbsDic":[str(round(mean_AbsCountDiff, 3))],  "std_AbsDic":[str(round(std_AbsCountDiff, 3))],
                             "mean_Agreement":[str(round(mean_CountAgreement, 3))],"std_Agreement":[str(round(std_CountAgreement, 3))],
                             "mean_MSE":[str(round(mean_MSE,3))], "std_MSE":[str(round(std_MSE, 3))],
                             "mean_R_2":[str(round(mean_R_2,3))], "std_R_2":[str(round(std_R_2, 3))],
                             "mean_ap": [str(mean_ap)], "std_ap": [str(std_ap)],
                             "all_dic": [['%.3f' % elem for elem in all_CountDiff]], "all_AbsDic": [['%.3f' % elem for elem in all_AbsCountDiff]],
                             "all_CountAgreement": [[ '%.3f' % elem for elem in all_CountAgreement]], "all_MSE": [[ '%.3f' % elem for elem in all_MSE]],
                             "all_R_2": [['%.3f' % elem for elem in all_R_2]], "all_ap": ap_data})

    df = df.append(new_data)
    df.to_csv(args.save_res_path,index=False)

    return mean_CountAgreement

def create_csv_files_for_fold(data_path, ds, cv_fold, exp_num, current_data_dict):


    # print relevant data to files
    csv_file_start = ds + '_cv' + str(cv_fold) + '_exp_'+ str(exp_num)
    csv_file_names = {}

    csv_file_names['train_count_file'] = os.path.join(data_path, csv_file_start + '_Train.csv')
    csv_file_names['train_centers_files'] = os.path.join(data_path, csv_file_start + '_Train_leaf_location.csv')

    csv_file_names['val_count_file'] = os.path.join(args.data_path, csv_file_start + '_Val.csv')
    csv_file_names['val_centers_files'] = os.path.join(args.data_path, csv_file_start + '_Val_leaf_location.csv')

    csv_file_names['test_count_file'] = os.path.join(args.data_path, csv_file_start + '_Test.csv')
    csv_file_names['test_centers_files'] = os.path.join(args.data_path, csv_file_start + '_Test_leaf_location.csv')

    # Remove files from prev runs if mistakenly exist
    for file in csv_file_names:
        if os.path.isfile(csv_file_names[file]):
            os.remove(csv_file_names[file])

    create_sub_csv_file(csv_file_names['train_count_file'],
                        csv_file_names['train_centers_files'],
                        current_data_dict['Train_leaf_counts'],
                        current_data_dict['Train_leaf_location_coord'], ds)

    create_sub_csv_file(csv_file_names['val_count_file'],
                        csv_file_names['val_centers_files'],
                        current_data_dict['Val_leaf_counts'],
                        current_data_dict['Val_leaf_location_coord'], ds)

    create_sub_csv_file(csv_file_names['test_count_file'],
                        csv_file_names['test_centers_files'],
                        current_data_dict['Test_leaf_counts'],
                        current_data_dict['Test_leaf_location_coord'], ds)

    return csv_file_names

def main(args=None):

    random.seed(50)
    np.random.seed(0)
    args.pipe =  'reg' #'reg' or 'keyPfinder'

    args.random_transform = True

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

    args.exp_name = 'reg_baseline_c5_dubreshko'

    args.lr = 1e-5
    args.reduce_lr = True
    args.reduceLR_patience = 5
    args.reduceLR_factor = 0.05

    args.early_stopping_indicator = "AbsCountDiff"
    args.early_stopping_patience = 50

    args.step_multi = 5

    args.multi_gpu = False
    args.multi_gpu_force = False

    if args.pipe == 'reg':
        args.option = args.exp_name
        args.calc_det_performance = False
        args.do_dropout = False

    elif args.pipe == 'keyPfinder':
        # key point detection options:
        # 10 - best option, as in the paper
        # 20 - reducing size GT Gaussian maps for the sub-model
        args.option = 20

        # the detection performance is done using the PCK metric - see our paper for mor information
        args.calc_det_performance = False

    else:
        print("Choose a relevant pipe - keyPfinder or reg")
        return

    args.save_res_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results", 'results_' + args.pipe + '_exp_'+ args.exp_name+ '_'+str(args.exp_num) + ".csv")

    images_num = {}
    images_num['A1'] = 128
    images_num['A2'] = 31
    images_num['A3'] = 27
    images_num['A4'] = 624
    images_num['BL'] = 1016

    # chosen_datasets = ['A1', 'A2', 'A3', 'A4']
    chosen_datasets = ['BL']

    num_of_CV = args.num_of_CV
    agreement_per_ds = {}
    total_num_of_images = 0
    total_mean_agreement = 0

    for ds in chosen_datasets:

        total_num_of_images += images_num[ds]

        if ds == 'A1' or ds == 'A2' or ds == 'A3' or ds == 'A4':
            args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                      'CVPPP2017_LCC_training', 'training', ds)
        elif ds == 'BL':
            args.data_path = os.path.join(GetEnvVar('DatasetsPath'), 'Counting Datasets',
                                          'Banana_leaves', ds)
        #
        # stats = {}
        # stats[ds] ={}
        #
        # stats[ds]['CountDiff'] = []
        # stats[ds]['AbsCountDiff'] = []
        # stats[ds]['CountAgreement'] = []
        # stats[ds]['MSE'] = []
        # stats[ds]['R_2'] = []
        # stats[ds]['ap'] = []

        All_Splitted_Data = data_split(args.data_path, ds)

        print('Working on dataset:', ds)

        if not os.path.isfile(args.save_res_path) :
            with open(args.save_res_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Exp", "Augmantation", "dataset", "dic", "AbsDic", "CountAgreement", "MSE",
                                 "R_2", "ap", "nd_weight","wd_weight", "epochs"])

        for cv_fold in range(1, 5):

            saving_path_name = os.path.join('exp_' + str(args.exp_num), 'cv_' + str(cv_fold))

            args.snapshot_path = os.path.join(GetEnvVar('ModelsPath'), 'Counting_Models_snapshots', args.pipe,
                                              saving_path_name)

            args.model = os.path.join(args.snapshot_path, 'resnet50_csv.h5')

            args.save_path = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, "results",
                                          saving_path_name)
            args.tensorboard_dir = os.path.join(GetEnvVar('ExpResultsPath'), 'Counting_Agri', args.pipe, 'log_dir',
                                                saving_path_name)

            Test_fold_num = (cv_fold) % num_of_CV+1
            Val_fold_num = (cv_fold + 1) % num_of_CV+1
            Train_fold_num = [(cv_fold + 2) % num_of_CV+1, (cv_fold + 3) % num_of_CV+1]

            current_data_dict = split_to_sets(All_Splitted_Data, Test_fold_num, Val_fold_num, Train_fold_num)
            csv_file_names = create_csv_files_for_fold(args.data_path, ds, cv_fold, args.exp_num, current_data_dict)

            args.train_csv_leaf_number_file = csv_file_names['train_count_file']
            args.train_csv_leaf_location_file = csv_file_names['train_centers_files']

            args.val_csv_leaf_number_file = csv_file_names['val_count_file']
            args.val_csv_leaf_location_file = csv_file_names['val_centers_files']

            # Train the model based on current split
            if args.pipe == 'keyPfinder':
                history = train.main(args)
            elif args.pipe == 'reg':
                train_reg.main(args)

            # Test the model

            # update args for evaluation
            args.val_csv_leaf_number_file = csv_file_names['test_count_file']
            args.val_csv_leaf_location_file = csv_file_names['test_centers_files']

            if args.calc_det_performance:
                CountDiff, AbsCountDiff, CountAgreement, MSE, R_2, ap = evaluate_LCC.main(args)
                ap = str(round(ap, 3))

            else:
                CountDiff, AbsCountDiff, CountAgreement, MSE, R_2 = evaluate_LCC.main(args)
                ap = 'not available'

            with open(args.save_res_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([str(args.exp_num),
                                 str(args.random_transform),
                                 ds,
                                 str(round(CountDiff, 3)),
                                 str(round(AbsCountDiff, 3)),
                                 str(round(CountAgreement, 3)),
                                 str(round(MSE, 3)),
                                 str(round(R_2, 3)),
                                 str(ap),
                                 str(args.epochs)])

            # print('Result of cv_',str(cv_fold),'-', 'testing ', ds)
            # print('CountDiff:',CountDiff, 'AbsCountDiff:', AbsCountDiff, 'CountAgreement', CountAgreement, 'MSE:', MSE)
            #

            # stats[ds]['CountDiff'].append(CountDiff)
            # stats[ds]['AbsCountDiff'].append(AbsCountDiff)
            # stats[ds]['CountAgreement'].append(CountAgreement)
            # stats[ds]['MSE'].append(MSE)
            # stats[ds]['R_2'].append(R_2)
            # stats[ds]['ap'].append(ap)

            # Delete current temp csv files
            for file in csv_file_names:
                if os.path.isfile(csv_file_names[file]):
                    os.remove(csv_file_names[file])

            if args.nd[3] == 1:
                break

        args.exp_num += 1
        # get mean and std errors, and save to results file

        # if not os.path.isfile(args.save_res_path) :
        #     with open(args.save_res_path, 'w', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow(["Exp", "Augmantation", "Train_set", "Test_set", "mean_Dic", "std_Dic",
        #                          "mean_AbsDic", "std_AbsDic", "mean_Agreement", "std_Agreement",
        #                          "mean_MSE", "std_MSE", 'mean_R_2', "std_R_2", "mean_ap", "std_ap",
        #                          "all_dic", "all_AbsDic", "all_CountAgreement", "all_MSE",
        #                          "all_R_2", "all_ap"])

    #     mean_CountAgreement = get_aggregated_results_withHyper(args, stats, ds)
    #     agreement_per_ds[ds] = mean_CountAgreement
    #
    # # get weighted average of count agreement
    # for ds in chosen_datasets:
    #     total_mean_agreement += agreement_per_ds[ds]*(images_num[ds]/total_num_of_images)
    #
    # print('total_mean_agreement:', total_mean_agreement)

    print("Done")

    return history


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

    args.random_transform = True
    args.evaluation = True
    # TODO - choose min and max image size
    args.image_min_side = 800
    args.image_max_side = 1333

    args.dataset_type = 'csv'

    args.exp_num = 1960
    args.gpu = '0'
    args.num_of_CV = 4

    args.epochs = 100
    args.num_of_CV = 4
    main(args)









