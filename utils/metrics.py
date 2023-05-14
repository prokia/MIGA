import scipy
import torch
import numpy as np
from scipy import interpolate, ndimage
import torchvision
import cv2
from datasets.utils.utils import recist_euclidean_distance
from scipy.spatial.distance import directed_hausdorff
import tqdm
from medpy import metric as medpyMetric

##########################################
# metric
##########################################
from utils.mask_operator import prob2mask


def mid_eval(pre_dict, gt_dict):
    mid = 0
    for item in gt_dict:
        pre_mask_list, gt_mask_path_list = pre_dict[item]['masks_list'], gt_dict[item]['mask_path_list']
        gt_mask_list = list(map(lambda x: cv2.imread(x, -1) // 255, gt_mask_path_list))
        gt, pre = np.zeros_like(gt_mask_list[0]), np.zeros_like(gt_mask_list[0])
        for gt_counters, gt_mask in enumerate(gt_mask_list):
            gt[gt_mask != 0] = gt_counters + 1
        for pre_counters, pre_mask in enumerate(pre_mask_list):
            pre[pre_mask != 0] = pre_counters + 1
        # gt_mask_list = list(map(mask2binary_mask, gt_mask_list))
        mi_dice = multi_inst_dice(gt, pre)
        mi_dice.pop('background')
        mi_dice = list(mi_dice.values())
        mean_mi_dice = sum(mi_dice) / len(mi_dice)
        mid += mean_mi_dice
    return mid / 210


def cal_dice(gt_seg, pred_seg, threshold=0.5):
    if pred_seg is None:
        pred_seg = np.zeros_like(gt_seg)
    pred_seg[pred_seg > threshold] = 1
    pred_seg[pred_seg <= threshold] = 0
    top = 2 * np.sum(gt_seg * pred_seg)
    bottom = np.sum(gt_seg) + np.sum(pred_seg)
    bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
    dicem = top / bottom
    return dicem


def cal_jaccard(gt_seg, pred_seg, threshold=0.5):
    if pred_seg is None:
        pred_seg = np.zeros_like(gt_seg)
    pred_seg = prob2mask(pred_seg, th=threshold)
    top = np.sum(gt_seg * pred_seg)
    bottom = np.sum(np.clip(gt_seg + pred_seg, 0, 1))
    bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
    j = top / bottom
    return j


def cal_foreground_pixels_precision(gt_seg, pred_seg, threshold=0.5):
    pred_seg = prob2mask(pred_seg, threshold)
    gt_seg = pred_seg * gt_seg
    return cal_jaccard(gt_seg, pred_seg)


def cal_foreground_pixels_recall(gt_seg, pred_seg, threshold=0.5):
    pred_seg = prob2mask(pred_seg, threshold)
    pred_seg = pred_seg * gt_seg
    return cal_jaccard(gt_seg, pred_seg)


def cal_pixels_precision(gt_seg, pred_seg, threshold=0.5):
    pred_seg = prob2mask(pred_seg, threshold)
    pred_nonzero = np.count_nonzero(pred_seg)
    if not pred_nonzero:
        return 1.
    correct = pred_seg & gt_seg
    precision = np.count_nonzero(correct) / pred_nonzero
    return precision


def cal_pixels_recall(gt_seg, pred_seg, threshold=0.5):
    pred_seg = prob2mask(pred_seg, threshold)
    gt_nonzero = np.count_nonzero(gt_seg)
    if not gt_nonzero:
        return 1.
    correct = pred_seg & gt_seg
    recall = np.count_nonzero(correct) / gt_nonzero
    return recall


def cal_dice_tensor(batch_gt_seg, batch_pred_seg, threshold=0.5):
    batch_pred_seg[batch_pred_seg > threshold] = 1
    batch_pred_seg[batch_pred_seg <= threshold] = 0
    top = 2 * (batch_gt_seg * batch_pred_seg).sum(dim=(-1, -2)).float()
    bottom = batch_gt_seg.sum(dim=(-1, -2)) + batch_pred_seg.sum(dim=(-1, -2)).float()
    bottom = torch.maximum(bottom, 1.0e-5 * torch.ones_like(bottom))  # add epsilon.
    dicem = top / bottom
    return dicem


# def cal_hausdorff(gt_seg, pred_seg, threshold=0.5):
#     if pred_seg is None:
#         pred_seg = np.zeros_like(gt_seg)
#     pred_seg[pred_seg > threshold] = 1
#     pred_seg[pred_seg <= threshold] = 0
#     hd = max(directed_hausdorff(gt_seg, pred_seg)[0], directed_hausdorff(pred_seg, gt_seg)[0])
#     return hd


def cal_hausdorff95(gt, pred_seg, threshold):
    from medpy.metric.binary import hd95 as medpy_hd95_function
    return medpy_hd95_function(gt, pred_seg)


def cal_hausdorff(gt, pred_seg, threshold):
    from medpy.metric.binary import hd as medpy_hd_function
    return medpy_hd_function(gt, pred_seg)


def cal_asd(gt, pred_seg, threshold):
    from medpy.metric.binary import asd as medpy_asd_function
    return medpy_asd_function(gt, pred_seg)


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if dim == 2:
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


# # HD95
# def cal_hausdorff95(s, g, spacing=None, threshold=0.5):
#     """
#     get the hausdorff distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s = prob2mask(s, threshold)
#     g = prob2mask(g, threshold)
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert image_dim == len(g.shape)
#     if spacing is None:
#         spacing = [1.0] * image_dim
#     else:
#         assert image_dim == len(spacing)
#     img = np.zeros_like(s)
#     if image_dim == 2:
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif image_dim == 3:
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)
#
#     dist_list1 = s_dis[g_edge > 0]
#     dist_list1 = sorted(dist_list1)
#     dist1 = dist_list1[int(len(dist_list1) * 0.95)]
#     dist_list2 = g_dis[s_edge > 0]
#     dist_list2 = sorted(dist_list2)
#     dist2 = dist_list2[int(len(dist_list2) * 0.95)]
#     return max(dist1, dist2)

# from sklearn.neighbors import KDTree
#
# def hd95(seg, gt):
#     struct = ndimage.generate_binary_structure(3, 1)
#
#     ref_border = gt ^ ndimage.binary_erosion(gt, struct, border_value=0)
#     ref_border_voxels = np.array(np.where(ref_border))  # 获取gt边界点的坐标,为一个n*dim的数组
#
#     seg_border = seg ^ ndimage.binary_erosion(seg, struct, border_value=0)
#     seg_border_voxels = np.array(np.where(seg_border)) # 获取seg边界点的坐标,为一个n*dim的数组
#
#     # 将边界点的坐标转换为实数值,单位一般为mm
#     ref_real = seg_border_voxels
#     gt_real = ref_border_voxels
#
#     tree_ref = KDTree(np.array(ref_border_voxels_real))
#     dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real, k=1)
#     tree_seg = KDTree(np.array(seg_border_voxels_real))
#     dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real, k=1)
#
#     hd = np.percentile(np.vstack((dist_seg_to_ref, dist_ref_to_seg)).ravel(), 95)
#
#     return hd

# def cal_hausdorff95(mask_gt, mask_pred, threshold=0.5, spacing=None):
#     mask_pred = prob2mask(mask_pred, threshold)
#     if np.sum(mask_pred) == 0:
#         # Follow Luo et al.
#         return 0
#     hd95 = medpyMetric.binary.hd95(mask_pred, mask_gt, voxelspacing=spacing)
#     return hd95

# def cal_hausdorff95(mask_gt, mask_pred, threshold=0.5, spacing=None):
#     return 0.95 * cal_hausdorff(mask_gt, mask_pred, threshold=threshold)


def margin_of_error_at_confidence_level(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


# # ASD
# def cal_asd(s, g, spacing=None):
#     """
#     get the average symetric surface distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert image_dim == len(g.shape)
#     if spacing is None:
#         spacing = [1.0] * image_dim
#     else:
#         assert image_dim == len(spacing)
#     img = np.zeros_like(s)
#     if image_dim == 2:
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif image_dim == 3:
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)
#
#     ns = s_edge.sum()
#     ng = g_edge.sum()
#     s_dis_g_edge = s_dis * g_edge
#     g_dis_s_edge = g_dis * s_edge
#     assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
#     return assd


# Multiple Instance Dice Similarity Coefficient
def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.
    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def multi_inst_dice(mask_gt, mask_pred):
    """Compute instance soerensen-dice coefficient.
        compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred` for multiple instances.
        Args:
          mask_gt: 3-dim Numpy array of type int. The ground truth image, where 0 means background and 1-N is an
                   instrument instance.
          mask_pred: 3-dim Numpy array of type int. The predicted mask, where 0 means background and 1-N is an
                   instrument instance.
        Returns:
          a instance dictionary with the dice coeffcient as float.
        """
    # get number of labels in image
    instances_gt = np.unique(mask_gt)
    instances_pred = np.unique(mask_pred)

    # create performance matrix
    performance_matrix = np.zeros((len(instances_gt), len(instances_pred)))
    masks = []

    # calculate dice score for each ground truth to predicted instance
    for counter_gt, instance_gt in enumerate(instances_gt):

        # create binary mask for current gt instance
        gt = mask_gt.copy()
        gt[mask_gt != instance_gt] = 0
        gt[mask_gt == instance_gt] = 1

        masks_row = []
        for counter_pred, instance_pred in enumerate(instances_pred):
            # make binary mask for current predicted instance
            prediction = mask_pred.copy()
            prediction[mask_pred != instance_pred] = 0
            prediction[mask_pred == instance_pred] = 1

            # calculate dice
            performance_matrix[counter_gt, counter_pred] = compute_dice_coefficient(gt, prediction)
            masks_row.append([gt, prediction])
        masks.append(masks_row)

    # assign instrument instances according to hungarian algorithm
    label_assignment = hungarian_algorithm(performance_matrix * -1)
    label_nr_gt, label_nr_pred = label_assignment

    # get performance per instance
    image_performance = []
    for i in range(len(label_nr_gt)):
        instance_dice = performance_matrix[label_nr_gt[i], label_nr_pred[i]]
        image_performance.append(instance_dice)

    missing_pred = np.absolute(len(instances_pred) - len(image_performance))
    missing_gt = np.absolute(len(instances_gt) - len(image_performance))
    n_missing = np.max([missing_gt, missing_pred])

    if n_missing > 0:
        for i in range(n_missing):
            image_performance.append(0)

    output = dict()
    for i, performance in enumerate(image_performance):
        if i > 0:
            output["tumor_{}".format(i - 1)] = performance
        else:
            output["background"] = performance

    return output


##########################################
# evaluate with FROC
##########################################
def IOU(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def FROC(boxes_all, gts_all, iou_th, nGt_all, nImg_all, fp_num):
    # nGt_all: total number of lesions
    # nImg_all: total number of images
    # Avoid the situation where the detection of an image is 0
    # In DeepLesion test set, nGt_all=4912, nImg_all=4817.

    # Compute the FROC curve, for single class only
    nImg = len(boxes_all)
    img_idxs = np.hstack([[i] * len(boxes_all[i]) for i in range(nImg)])
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    scores = scores[ord]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])  # compare one box with all gts in one image
        if overlaps.max() < iou_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)

    # nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt_all
    fp_per_img = np.array(fps, dtype=float) / nImg_all
    score = scores[np.where(fp_per_img <= fp_num)].min()

    return sens, fp_per_img, score


def sens_at_FP(boxes_all, gts_all, nGt, nImg, avgFP=[0.5, 1, 2, 4, 8, 16], iou_th=0.5, fp_num=0.5):
    # compute the sensitivity at avgFP (average FP per image)
    if not boxes_all:
        return np.array([0.] * len(avgFP)), 0., [0.]
    sens, fp_per_img, score = FROC(boxes_all, gts_all, iou_th, nGt, nImg, fp_num)
    f = interpolate.interp1d(fp_per_img, sens, kind='nearest', fill_value='extrapolate')
    res = f(np.array(avgFP))
    return res, score, fp_per_img


# #################
# Eval RECIST
# #################
def recist_eval(pre_dict, gt_dict, logger=None, mode='cross'):
    long_diff, short_diff = [], []
    for item in gt_dict:
        if item not in pre_dict:
            continue
        pre_pts_list, score_list = pre_dict[item]['extreme_points_list'], pre_dict[item]['score_list']
        spacing = pre_dict[item]['img_spacing']
        choose_pts, choose_score = None, 0
        for pts, score in zip(pre_pts_list, score_list):
            if score > choose_score:
                choose_pts, choose_score = pts.copy(), score
        gt_pts = gt_dict[item]['extreme_points_list'][0]
        if mode == 'cross':
            pre_dia = [recist_euclidean_distance(choose_pts[1], choose_pts[3], spacing),
                       recist_euclidean_distance(choose_pts[0], choose_pts[2], spacing)]
        else:
            pre_dia = [recist_euclidean_distance(choose_pts[0], choose_pts[1], spacing),
                       recist_euclidean_distance(choose_pts[2], choose_pts[3], spacing)]
        pre_dia.sort()
        gt_dia = [recist_euclidean_distance(gt_pts[0], gt_pts[2], spacing),
                  recist_euclidean_distance(gt_pts[1], gt_pts[3], spacing)]
        gt_dia.sort()
        Ldiff, Sdiff = abs(pre_dia[1] - gt_dia[1]), abs(pre_dia[0] - gt_dia[0])
        long_diff.append(Ldiff)
        short_diff.append(Sdiff)
        if logger is not None:
            logger.info(f'>> {item} Ldiff: {Ldiff} Sdiff: {Sdiff}')
            print(f'{item} Ldiff: {Ldiff} Sdiff: {Sdiff}')

    long_diff, short_diff = np.array(long_diff), np.array(short_diff)
    long_diff_mean, short_diff_mean = np.mean(long_diff), np.mean(short_diff)
    long_diff_std, short_diff_std = np.std(long_diff), np.std(short_diff)
    return long_diff_mean, short_diff_mean, long_diff_std, short_diff_std
