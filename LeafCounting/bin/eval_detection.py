
import cv2
import numpy as np

from ..utils.read_activations import get_activations

def detection_evaluation(image_name, model, image, GT_centers, alpha=0.1):
    local_soft_max_activations = get_activations(model, model_inputs=image[0], print_shape_only=False,
                                                 layer_name='smooth_step_function2')
    local_soft_max_activations = local_soft_max_activations[0][0, :, :, 0]
    w_plant, h_plant = extract_plant_BB(image_name, local_soft_max_activations)
    detections_map = np.where(local_soft_max_activations > 0.02, local_soft_max_activations, 0)
    GT_centers = np.where(GT_centers == 1, GT_centers, 0)
    Y_detect, X_detect = np.nonzero(detections_map)
    Y_GT, X_GT = np.nonzero(GT_centers)
    detection_scores = detections_map[Y_detect, X_detect]
    detections = np.array([Y_detect,X_detect, detection_scores])
    sorted_detections = detections[:, (detections[2, :]*-1).argsort()]

    reduced_GT_centers = np.array([Y_GT, X_GT])
    pck_val_thresh = alpha*np.max([w_plant, h_plant])
    t=[]
    p=[]
    for det_num in range(sorted_detections.shape[1]):
        det = sorted_detections[[0,1], det_num]
        dists = np.sqrt(np.sum(np.array([reduced_GT_centers[:, i] - det for i in range(reduced_GT_centers.shape[1])]) ** 2, axis=-1))
        closest_GT_ind = np.argmin(dists)
        min_dist = np.min(dists)
        if reduced_GT_centers.shape[1] == 0:
            break
        if(min_dist <= pck_val_thresh):
            p.append(sorted_detections[-1, det_num])
            t.append(1)
            reduced_GT_centers = np.delete(reduced_GT_centers, closest_GT_ind, 1)
        else:
            p.append(sorted_detections[-1, det_num])
            t.append(0)

    if len(reduced_GT_centers):
        for i in range(reduced_GT_centers.shape[1]):
            p.append(0)
            t.append(1)

    return t, p

def extract_plant_BB(image_name, activation_map):
    image_shape = activation_map.shape
    mask_image_path = image_name + '_fg.png'
    plant_mask_image = cv2.imread(mask_image_path, 0)

    plant_mask_image = cv2.resize(plant_mask_image, (image_shape[1], image_shape[0]))
    Ys, Xs = np.nonzero(plant_mask_image)
    y_min = np.min(Ys)
    y_max = np.max(Ys)
    x_min = np.min(Xs)
    x_max = np.max(Xs)

    return (x_max - x_min),(y_max - y_min)

def calc_recall_precision_ap(T,P):
    P = np.array(P)
    T = np.array(T)
    npos = np.sum(T)
    # clean zerows (that represent true negative)
    T = T[np.where(P > 0)]
    P = P[np.where(P > 0)]
    # sort by confidence
    sorted_ind = np.argsort(-P)
    P = P[sorted_ind]
    T = T[sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(P)
    if nd > 0:
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for i in np.arange(nd):
            if T[i] == 1 and P[i] > 0:
                tp[i] = 1.
            elif T[i] == 0 and P[i] > 0:
                fp[i] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / float(npos)
        precision = tp / (tp + fp)
    else:
        recall = [0]
        precision = [0]

    ap = measure_ap(recall, precision)

    return recall, precision, ap

def measure_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

