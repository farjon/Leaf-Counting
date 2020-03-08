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

from __future__ import print_function

from .read_activations import get_activations
from ..bin.eval_detection import detection_evaluation, calc_recall_precision_ap
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import sys
import argparse


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
#import keras_retinanet.bin
__package__ = "keras_retinanet.bin"


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')

    return parser.parse_args(args)


def visualize_images(output, Image_name, save_path, generator, model, image):
    if not generator.epoch == None:
        current_epoch = str(generator.epoch+1) #generator.epoch in [0,99]
    else:
        current_epoch = 'test'

    visualization_path = os.path.join(save_path, 'epoch_' + current_epoch)

    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    fgImage_name = Image_name + "_fg.png"

    # Draw GT activations:
    #plt.figure()
    #plt.axis("off")
    background = Image.open(os.path.join(generator.base_dir, fgImage_name), 'r')
    background = background.convert("RGBA")
    BG_w, BG_h = background.size
    # plt.imshow(background)
    # plt.savefig(visualization_path + '/' + Image_name + '_BG.png', pad_inches=0)  # transparent=True,
    # plt.close()

    anno = output[2][0, :, :, 0]
    plt.figure()
    #heat_map = seaborn.heatmap(anno, xticklabels=False, yticklabels=False, cbar=False)
    #heat_map = heat_map.despine
    #heat_map = heat_map.get_figure()
    plt.imshow(anno)
    plt.imsave(visualization_path + '/' + Image_name + '_anno.png', anno)
    gt_anns = Image.open(visualization_path + '/' + Image_name + '_anno.png')
    gt_anns = gt_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
    plt.imsave(visualization_path + '/' + Image_name + '_anno.png', gt_anns)
    plt.close()

    # out = image1 * (1.0 - alpha) + image2 * alpha
    plt.figure()
    plt.axis("off")
    alphaBlended = Image.blend(gt_anns, background, 0.6)
    plt.imshow(alphaBlended)
    plt.imsave(visualization_path + '/' + Image_name + '_Blended_GT.png',alphaBlended )
    plt.close()

    # Relu map #######################################################################################################
    plt.figure()
    classification_submodel_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                                          layer_name='pyramid_classification_relu')
    classification_submodel_activations = classification_submodel_activations[0][0, :, :, 0]

    plt.imshow(classification_submodel_activations)
    plt.imsave(visualization_path + '/' + Image_name + '_Relu.png', classification_submodel_activations)
    relu_anns = Image.open(visualization_path + '/' + Image_name + '_Relu.png')

    #relu_anns = relu_anns.convert("RGBA")
    relu_anns = relu_anns.resize((BG_w, BG_h))  # Image.ANTIALIAS
    plt.imsave(visualization_path + '/' + Image_name + '_Relu.png', relu_anns)
    plt.close()

    plt.figure()
    plt.axis("off")
    alphaBlended_relu = Image.blend(relu_anns, background, 0.6)
    plt.imshow(alphaBlended_relu)
    plt.imsave(visualization_path + '/' + Image_name + '_Blended_Relu.png', alphaBlended_relu)
    plt.close()

    # softmax map #####################################################################################################

    plt.figure()
    local_soft_max_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                                 layer_name='LocalSoftMax')
    local_soft_max_activations = local_soft_max_activations[0][0, :, :, 0]

    plt.imshow(local_soft_max_activations)
    plt.imsave(visualization_path + '/' + Image_name + '_softmax.png', local_soft_max_activations)
    softmax_anns = Image.open(visualization_path + '/' + Image_name + '_softmax.png')

    #softmax_anns = softmax_anns.convert("RGBA")
    softmax_anns = softmax_anns.resize((BG_w, BG_h))  # Image.
    plt.imsave(visualization_path + '/' + Image_name + '_softmax.png', softmax_anns)
    plt.close()

    plt.figure()
    plt.axis("off")
    alphaBlended_softmax = Image.blend(softmax_anns, background, 0.6)
    plt.imshow(alphaBlended_softmax)
    plt.imsave(visualization_path + '/' + Image_name + '_Blended_softmax.png', alphaBlended_softmax)
    plt.close()

    '''
    #################################################################################################
    plt.figure()
    anno = output[1][0,:,:,0]
    img = seaborn.heatmap(anno)
    plt.title("GT_Gaussians: "+Image_name)
    plt.savefig(save_path+'/'+Image_name+'_anno.png')

    plt.figure()
    classification_submodel_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                                          layer_name='pyramid_classification_relu')
    display_activations(classification_submodel_activations)
    plt.title("Last Relu map : "+rgbImage_name)

    plt.figure()
    local_soft_max_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                         layer_name='LocalSoftMax')
    display_activations(local_soft_max_activations)
     plt.title("LocalSoftMax map : "+rgbImage_name)

    sigmoid_sum = np.sum(classification_submodel_activations)
    softmax_sum = np.sum(local_soft_max_activations)
    '''
    #plt.figure()
    # plt.axis("off")
    # anno = output[1][0, :, :, 0]
    # gt_anns = seaborn.heatmap(anno) #xticklabels=False, yticklabels=False, cbar=False
    # plt.savefig(save_path + '/' + Image_name + '_anno.png', pad_inches=0) #transparent=True
    # img1 = Image.open(save_path + '/' + Image_name + '_anno.png') #Image._conv_type_shape(), 480,640,4


def _get_GTandPredictions(option, counts_file, generator, model, save_path=None, calc_det_performance = False):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_GT_counts[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        option          : The option number of the model
        counts_file     : The ground truth CSV data file
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """

    visualize_im = False
    all_GT_counts = []
    all_predicted_counts = []
    alpha = 0.1
    T, P = [], []
    for index in range(generator.size()):
        image, output = generator.next()

        image_index = generator.groups[generator.group_index-1][0]
        full_rgbImage_name = generator.rbg_images_names[image_index]
        Image_name = full_rgbImage_name.split("_rgb")[0]

        if visualize_im:
            if not generator.epoch == None:
                if generator.epoch==0 or (generator.epoch+1) % 20 == 0 :
                    visualize_images(output, Image_name, save_path, generator, model, image)
            else:
                visualize_images(output, Image_name, save_path, generator, model, image)


        # get GT predictions
        all_GT_counts.append(output[0][0])

        if calc_det_performance:
            t, p = detection_evaluation(os.path.join(generator.base_dir, Image_name), model, image, output[-1][0,:,:,0], alpha)
            T = T + t
            P = P + p

        # get predictions - run network
        if option == 'reg_baseline_c5_dubreshko' or option == 'reg_baseline_c5' or option == 'reg_fpn_p3' or \
                option == 'reg_fpn_p3_p7_avg' or option == 'reg_fpn_p3_p7_mle' or option == 'reg_fpn_p3_p7_min_sig' \
                or option == 'reg_fpn_p3_p7_min_sig_L1' or option == 'reg_fpn_p3_p7_mle_L1':
            count = model.predict_on_batch(image)
        if option == 'reg_baseline_c5_dubreshko' or option == 'reg_baseline_c5' or option == 'reg_fpn_p3':
            count = count[0][0]
        if option == 'reg_fpn_p3_p7_mle':
            count = count[0]
        if option == 'reg_fpn_p3_p7_avg':
            mus = [count[i][0][0] for i in range(len(count))]
            count = np.mean(mus)
            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

        if option == 'reg_fpn_p3_p7_min_sig' or option == 'reg_fpn_p3_p7_min_sig_L1':
            mus = [count[i][0][0] for i in range(len(count))]
            sigmas = [count[i][0][1] for i in range(len(count))]
            count = mus[np.argmin(sigmas)]
            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

        if option == 'reg_fpn_p3_p7_mle_L1':
            mus = np.asarray([count[i][0][0] for i in range(len(count))])
            sigmas = np.asarray([1/np.exp(count[i][0][1]) for i in range(len(count))])
            sorted_inds = np.argsort(mus)
            mus = mus[sorted_inds]
            sigmas = sigmas[sorted_inds]
            procesed_sigmas = np.cumsum(sigmas)/np.sum(sigmas)
            mle_ind = np.where(procesed_sigmas > 0.5)[0]
            count = mus[mle_ind][0]

            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

            # count_best = mus[np.argmin(np.abs([x-output[0][0] for x in mus]))]
            #print("image:", image_name, "GT:", current_GT, ",", "predicted:", round(count), "(", count,")","sigma: ", str(np.exp(0.5*log_var)),"log(var): ", str(log_var))
            #print("GT:", output[0][0], ",", "count_best:", count) #,' mle:',mle, " count: ", count ," count_mean: ",count_mean,"sigma: ", str(np.exp(0.5*log_var)),"log(var): ", str(log_var))

        # elif option == 1:
        #
        #     from .. import losses
        #     import keras
        #     # vriable_losses = {'pyramid_classification_relu': losses.focal_gyf(),'SumPooling_cls_output': keras.losses.mae}
        #     # compile model
        #     # model.compile(
        #     #      loss=vriable_losses,
        #     #      optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))
        #
        #     count = model.predict_on_batch(image)[0]
        #
        #     # anno = generator.compute_keypoints_targets(image.shape ,[np.array(center_coordinates)])
        #     # score = model.evaluate(np.expand_dims(image, axis=0), [np.expand_dims(current_GT, axis=0),np.expand_dims(np.expand_dims(anno, axis=0), axis=3)], verbose=0)
        #     # print(model.evaluate_generator(generator, steps = int(generator.size()), verbose = 1))
        #     # print("image:", image_name, "GT:", current_GT, ",", "predicted:", round(count[0][0]), "(", count[0][0], ")", ", abs diff: ", round(abs(current_GT - count[0][0]), 2) )
        #     print("GT:", output[0][0], ",", "predicted:", round(count[0][0]), "(", count[0][0], ")",
        #           ", abs diff: ", round(abs(output[0][0] - count[0][0]), 2))

        elif option == 2 or option == 3 or option == 10 or option == 1 or option == 20:
            count = model.predict_on_batch(image)[0][0][0]
            #
            print("image:", Image_name, "GT:", output[0][0], ",", "predicted:", round(count), "(", count, ")",
                  ", abs diff: ", round(abs(output[0][0] - count), 2))

        count = round(count)
        all_predicted_counts.append(count)



    if calc_det_performance:
        recall, precision, ap = calc_recall_precision_ap(T, P)
        plot_RP_curve(recall, precision, ap, save_path)
        return all_GT_counts, all_predicted_counts, ap

    return all_GT_counts, all_predicted_counts

def plot_RP_curve(recall, precision, ap, save_path):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.99)
    plt.fill_between(recall, precision, step='post', color='b', alpha=0.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_path = os.path.join(save_path + '\\RP_curve.png')
    plt.savefig(plot_path)
    plt.close(plot_path)


def _get_prediction(option, generator, model, save_path='', calc_det_performance=False):

    visualize_im = False
    all_predicted_counts = []

    for index in range(generator.size()):
        # this is not the real cond, it needs to check if we are in test/train mode
        if visualize_im == True:
            image, output = generator.next()
        else:
            image = generator.next()
        image_index = generator.groups[generator.group_index-1][0]
        full_rgbImage_name = generator.rbg_images_names[image_index]
        Image_name = full_rgbImage_name.split("_rgb")[0]

        if visualize_im:
            if not generator.epoch == None:
                if generator.epoch==0 or (generator.epoch+1) % 20 == 0 :
                    visualize_images(output, Image_name, save_path, generator, model, image)
            else:
                visualize_images(output, Image_name, save_path, generator, model, image)


        if option == 'reg_baseline_c5_dubreshko' or option == 'reg_baseline_c5' or option == 'reg_fpn_p3' or \
                option == 'reg_fpn_p3_p7_avg' or option == 'reg_fpn_p3_p7_mle' or option == 'reg_fpn_p3_p7_min_sig' \
                or option == 'reg_fpn_p3_p7_min_sig_L1' or option == 'reg_fpn_p3_p7_mle_L1':
            count = model.predict_on_batch(image)
        if option == 'reg_baseline_c5_dubreshko' or option == 'reg_baseline_c5' or option == 'reg_fpn_p3':
            count = count[0][0]
        if option == 'reg_fpn_p3_p7_mle':
            count = count[0]
        if option == 'reg_fpn_p3_p7_avg':
            mus = [count[i][0][0] for i in range(len(count))]
            count = np.mean(mus)
            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

        if option == 'reg_fpn_p3_p7_min_sig' or option == 'reg_fpn_p3_p7_min_sig_L1':
            mus = [count[i][0][0] for i in range(len(count))]
            sigmas = [count[i][0][1] for i in range(len(count))]
            count = mus[np.argmin(sigmas)]
            print("image:", Image_name, "GT:", output[0][0], ", predictions:", mus, ", count: ", count)

        if option == 'reg_fpn_p3_p7_mle_L1':
            mus = np.asarray([count[i][0][0] for i in range(len(count))])
            sigmas = np.asarray([1 / np.exp(count[i][0][1]) for i in range(len(count))])
            sorted_inds = np.argsort(mus)
            mus = mus[sorted_inds]
            sigmas = sigmas[sorted_inds]
            procesed_sigmas = np.cumsum(sigmas) / np.sum(sigmas)
            mle_ind = np.where(procesed_sigmas > 0.5)[0]
            count = mus[mle_ind][0]
        elif option == 2 or option == 3 or option == 10 or option == 1 or option == 20:
            count = model.predict_on_batch(image)[0][0][0]
            #
        print("image:", Image_name, ",", "predicted:", round(count), "(", count, ")",
                  ", ")

        count = round(count)
        all_predicted_counts.append(count)

    return all_predicted_counts


def SumOfDifferences(A, B):
    # calculate sum of A - B
    #  A and B must have the same size
    out = 0
    for i in range(len(A)):
        out+=(A[i]-B[i])
    return out


def SumOfAbsDifferences(A,B):
    # calculate sum of |A-B|
    # A and B must have the same size

    out = 0
    for i in range(len(A)):
        out += abs(A[i] - B[i])

    return out

def SumAgreement(A,B):
    # calculate sum A[i] == B[i]
    # A and B must have the same size

    out = 0
    for i in range(len(A)):
        if A[i] == B[i]:
            out += 1

    return out

def evaluate(option, counts_file, generator,model, save_path=None, calc_det_performance = False):

        """ Evaluate a given dataset using a given model.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """
        # gather all detections and annotations
        if calc_det_performance:
            all_GT_counts, all_predicted_counts, ap = _get_GTandPredictions(option, counts_file, generator, model,
                                                                        save_path=save_path,
                                                                        calc_det_performance=calc_det_performance)
        else:
            all_GT_counts, all_predicted_counts = _get_GTandPredictions(option, counts_file, generator, model,
                                                                            save_path=save_path,
                                                                            calc_det_performance=calc_det_performance)


        average_precisions = {}

        # num_of_images = len(all_GT_counts)
        # CountDiff = SumOfDifferences(all_GT_counts, all_predicted_counts)/num_of_images
        # AbsCountDiff = SumOfAbsDifferences(all_GT_counts, all_predicted_counts)/num_of_images
        # CountAgreement  = SumAgreement(all_GT_counts, all_predicted_counts)/num_of_images

        all_GT_counts = np.array(all_GT_counts)
        all_predicted_counts = np.array(all_predicted_counts)

        CountDiff = np.mean(all_GT_counts - all_predicted_counts)
        AbsCountDiff = np.mean(np.abs(all_GT_counts - all_predicted_counts))
        MSE = np.mean((all_GT_counts - all_predicted_counts)**2)
        CountAgreement = np.mean(np.sum(np.where(all_GT_counts == all_predicted_counts)))

        if calc_det_performance:
            return CountDiff, AbsCountDiff, CountAgreement , MSE, ap

        return CountDiff, AbsCountDiff, CountAgreement, MSE


