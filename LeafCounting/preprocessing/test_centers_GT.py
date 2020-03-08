import os
import sys
import argparse
import GetEnvVar as env
import cv2
import numpy as np
import csv




def draw_center_points(data):

    with open(rgb_counts_file) as csvfile:
        readCSV = csv.reader(csvfile)
        count = 0
        for row in readCSV:
            #print(row)
            rgbImage_name = row[0]
            plant_name = rgbImage_name.split("_")[0]

            rgb_image_1 = cv2.imread(os.path.join(dataset_path, rgbImage_name))
            centers_image = cv2.imread(os.path.join(dataset_path, plant_name+"_centers.png"), 0)

            ################################################################################
            if data != "A4":
                Ys, Xs= np.nonzero(centers_image)
            else:
                Xs, Ys = np.nonzero(centers_image)

            ################################################################################

            numb_of_center_points = len(Xs)
            GT_leaves_count = int(row[1])
            #print(numb_of_center_points, GT_leaves_count)

            if numb_of_center_points != GT_leaves_count:
                count+=1


            for i in range(numb_of_center_points):
                cy, cx = Ys[i], Xs[i]
                cv2.circle(rgb_image_1,(cx, cy), 5, (0, 0, 255), 1)

                # cv2.imshow("",rgb_image_1)
                # cv2.waitKey(0)

            cv2.imwrite(os.path.join(output_path_centers, data+ "_" + plant_name + "_with_centers.png") ,rgb_image_1)


    print("Dataset", data, ": Num of differenr counts in files vs. center point is:", count)

def draw_plant_mask(data):
    with open(rgb_counts_file) as csvfile:
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            #print(row)
            rgbImage_name = row[0]
            plant_name = rgbImage_name.split("_")[0]
            rgb_image_2 = cv2.imread(os.path.join(dataset_path, rgbImage_name))
            mask_image = cv2.imread(os.path.join(dataset_path, plant_name + "_fg.png"))

            Ys2, Xs2, z = np.nonzero(mask_image)

            for i in range(len(Xs2)):
                cy2, cx2 = Ys2[i], Xs2[i]
                cv2.circle(rgb_image_2, (cx2, cy2), 5, (0, 0, 255), 1)

            # cv2.imshow(plant_name,rgb_image_2)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(output_path_masks, data + "_" + plant_name + "_with_mask.png"), rgb_image_2)

def draw_points_from_centers_csv(data):

    with open(leaf_points_file) as csvfile:
        readCSV = csv.reader(csvfile)

        points_dict = {}
        for row in readCSV:
            #print(row)
            key = row[0]
            if len(points_dict)==0:
                points_dict[key] = []
                points_dict[key].append([row[1], row[2]])
            else:
                if key in points_dict.keys():
                    points_dict[key].append([row[1], row[2]])
                else:
                    points_dict[key] = []
                    points_dict[key].append([row[1], row[2]])


    for key in points_dict.keys():
        image_name= key
        plant_name = image_name.split("_")[0]
        rgb_image_3 = cv2.imread(os.path.join(dataset_path, plant_name+"_rgb.png"))

        points = points_dict[image_name]

        for i in range(len(points)):
            cx = int(points[i][0])
            cy = int(points[i][1])
            cv2.circle(rgb_image_3,(cx, cy), 5, (0, 0, 255), 1)


        cv2.imwrite(os.path.join(output_path_pointsFromFile, data+ "_" + plant_name + "_centersFromFile.png") ,rgb_image_3)



if __name__ == "__main__":

    datapath = os.path.join(env.GetEnvVar('DatasetsPath'), 'Phenotyping Datasets', 'Plant phenotyping','CVPPP2017_LCC_training', 'training')


    dataset_name = ["A1", "A2", "A3",] #['A1', "A2", "A3", "A4"]

    for i in range(len(dataset_name)):
        data = dataset_name[i]
        dataset_path = os.path.join(datapath, data)

        output_path_centers = os.path.join(datapath, data, "images_with centers")
        output_path_masks = os.path.join(datapath, data, "images_with masks")
        output_path_pointsFromFile = os.path.join(datapath, data, "images_with centers from files")

        if not os.path.exists(output_path_centers):
            os.makedirs(output_path_centers)

        if not os.path.exists(output_path_masks):
            os.makedirs(output_path_masks)

        if not os.path.exists(output_path_pointsFromFile):
            os.makedirs(output_path_pointsFromFile)

        rgb_counts_file = os.path.join(dataset_path, data + ".csv")
        leaf_points_file = os.path.join(dataset_path, data + "_leaf_location.csv")


        draw_center_points(data)
        draw_plant_mask(data)

        draw_points_from_centers_csv(data)
        print("Done with " , data)





