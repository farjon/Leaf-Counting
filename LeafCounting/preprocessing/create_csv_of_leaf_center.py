import csv
import cv2
import os
import numpy as np
from GetEnvVar import GetEnvVar

'''
This script generates a csv file that contains the coordinates of the leaves center points, as provided by 
the XXX_centers.gpg images within a given dataset directory.

Thw main function gets as input:
Ai_data_path - the path to the relevant dataset directory
dataset_name -  the specific dataset name

'''


def find_images_names(csv_file_name):
    masks_center_names = []
    with open(csv_file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            image_name = row[0]
            plant_number = image_name[0:image_name.find('rgb')]
            mask_name = plant_number + 'centers.png'
            masks_center_names.append(mask_name)

    return masks_center_names

def create_data_to_write(Ai_data_path, masks_center_names, dataset_name):
    new_csv_file_data = []
    for mask_name in masks_center_names:
        print(mask_name)
        mask_path = os.path.join(Ai_data_path, mask_name)
        # read the mask in gray scale
        mask = cv2.imread(mask_path, 0)
        ######################################################################

        if dataset_name!="A4":
            Ys, Xs = np.nonzero(mask)
        else:
            Xs, Ys  = np.nonzero(mask)
        ######################################################################

        for index in range(len(Xs)):
            x = Xs[index]
            y = Ys[index]
            line = [mask_name, x, y]
            new_csv_file_data.append(line)

    return new_csv_file_data

def write_to_csv(new_csv_file_name, new_csv_file_data):
    with open(new_csv_file_name, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for row in new_csv_file_data:
            wr.writerow([row[0], row[1], row[2]])

def main(Ai_data_path, dataset_name):

    print('creating leaf center csv on dataset {}'.format(dataset_name))

    # find images names
    csv_file_name = os.path.join(Ai_data_path, dataset_name + '.csv')
    masks_center_names = find_images_names(csv_file_name)
    centers_file_path = os.path.join(Ai_data_path, dataset_name + "_leaf_location.csv")

    new_csv_file_data = create_data_to_write(Ai_data_path, masks_center_names, dataset_name)

    # write the data
    write_to_csv(centers_file_path, new_csv_file_data)

if __name__ == "__main__":

    dataset_name = "A4"
    plant_phen_path = os.path.join(GetEnvVar('DatasetsPath'), 'Phenotyping Datasets', 'Plant phenotyping')
    data_path = os.path.join(plant_phen_path, 'CVPPP2017_LCC_training', 'training')
    Ai_data_path = os.path.join(data_path, dataset_name)

    main(Ai_data_path, dataset_name)