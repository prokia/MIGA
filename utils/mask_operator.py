import cv2
import torch
from skimage import measure
import numpy as np
from scipy.ndimage import binary_fill_holes
from utils.geometry import close_contour, find_recist_in_contour, recist2det
from visualizer.visualizers import vis_img, plt_show
from .bbox_operator import BBoxes
from functools import reduce


def read_mask_from_path_list(path_list):
    re = []
    for path in path_list:
        mask = cv2.imread(path, -1)
        mask[mask != 0] = 1
        re.append(mask)
    return re


def mask_union(masks, height=None, width=None):
    if isinstance(masks, list):
        if masks:
            mask = sum(masks)
        else:
            if height is not None:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                mask = None
    else:
        mask = masks.copy()
    if mask is not None:
        mask[mask != 0] = 1
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    return mask


def mask2FBmask(mask):
    mask = mask.copy()
    mask[mask != 0] = 1
    mask[mask != 1] = 0
    return mask


def excluede_irrelevant_mask_region(mask, keep_value):
    mask = mask.copy()
    mask[mask != keep_value] = 0
    mask[mask != 0] = 1
    return mask


def mask2det(mask, expand=5, fill_holes=True, boxize=False):
    if fill_holes:
        mask = binary_fill_holes(mask, structure=None, output=None, origin=0)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox_list = []
    for contour in contours:
        x_min, y_min = np.min(contour[:, 0, 0]), np.min(contour[:, 0, 1])
        x_max, y_max = np.max(contour[:, 0, 0]), np.max(contour[:, 0, 1])
        x_min, y_min = max(0, x_min - expand), max(0, y_min - expand)
        x_max, y_max = min(w, x_max + expand), min(h, y_max + expand)
        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox_list.append(bbox)
    re = BBoxes(bbox_list, mode='xyxy') if boxize else np.array(bbox_list)
    return re


def mask2det2RECIST(mask, expand=5, fill_holes=True, boxize=False):
    if fill_holes:
        mask = binary_fill_holes(mask, structure=None, output=None, origin=0)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox_list = []
    if fill_holes:
        mask = binary_fill_holes(mask, structure=None, output=None, origin=0)
    recist_pts = np.expand_dims(get_recist_from_crop_mask(mask), axis=0)
    for contour in contours:
        x_min, y_min = np.min(contour[:, 0, 0]), np.min(contour[:, 0, 1])
        x_max, y_max = np.max(contour[:, 0, 0]), np.max(contour[:, 0, 1])
        x_min, y_min = max(0, x_min - expand), max(0, y_min - expand)
        x_max, y_max = min(w, x_max + expand), min(h, y_max + expand)
        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox_list.append(bbox)
    re = BBoxes(bbox_list, mode='xyxy') if boxize else np.array(bbox_list)
    return re


def recist2recist_channel(recist_pts_list, shape, thickness=0):
    recist_channel = np.zeros(shape)
    for recist_pts in recist_pts_list:
        recist_pts = np.array(recist_pts).astype(np.int).reshape((4, 2))
        recist_tuple = tuple(map(tuple, recist_pts))
        cv2.line(recist_channel, recist_tuple[0], recist_tuple[1], 1, thickness)
        cv2.line(recist_channel, recist_tuple[2], recist_tuple[3], 1, thickness)
    return recist_channel.astype(np.uint8)


def mask2Globaldet(mask, expand=5, fill_holes=True, boxize=True):
    if fill_holes:
        mask = binary_fill_holes(mask, structure=None, output=None, origin=0)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    global_x_min, global_y_min = w - 1, h - 1
    global_x_max, global_y_max = 0, 0
    for contour in contours:
        x_min, y_min = np.min(contour[:, 0, 0]), np.min(contour[:, 0, 1])
        x_max, y_max = np.max(contour[:, 0, 0]), np.max(contour[:, 0, 1])
        x_min, y_min = max(0, x_min - expand), max(0, y_min - expand)
        x_max, y_max = min(w - 1, x_max + expand), min(h - 1, y_max + expand)
        global_x_min, global_y_min = min(x_min, global_x_min), min(y_min, global_y_min)
        global_x_max, global_y_max = max(x_max, global_x_max), max(global_y_max, y_max)
    bbox = [global_x_min, global_y_min, global_x_max, global_y_max]
    re = BBoxes(bbox, mode='xyxy') if boxize else np.array(bbox)
    return re


def mask2RECIST2det(mask, expand=5, fill_holes=True, height=512, width=512):
    if fill_holes:
        mask = binary_fill_holes(mask, structure=None, output=None, origin=0)
    recist_pts = np.expand_dims(get_recist_from_crop_mask(mask), axis=0)
    bbox = recist2det(recist_pts, expand, height, width)
    return bbox, recist_pts


def largest_connect_component(mask):
    labeled_img, num = measure.label(mask, background=0, return_num=True, connectivity=1)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        num = np.sum(labeled_img == i)
        if num > max_num:
            max_num = num
            max_label = i
    mcr = (labeled_img == max_label)
    return mcr


def post_process_from_diamond(diamond, raw_mask, image, th=120):
    # post_mask = raw_mask.copy()
    image = np.expand_dims(image, axis=0)
    iner = image * diamond
    mean_pix = np.sum(iner) / np.sum(diamond)
    th0 = max(abs(iner.min() - mean_pix), abs(iner.max() - mean_pix))
    clip0 = np.abs(image - mean_pix) > th0
    clip = np.abs(image - mean_pix) > th
    clip = clip0 + clip
    zero = np.zeros_like(raw_mask)
    post_mask = np.where(clip, zero, raw_mask)
    post_mask[0] = cv2.erode(post_mask[0], kernel=(3, 3), iterations=1)
    post_mask[0] = largest_connect_component(post_mask[0])
    post_mask[0] = binary_fill_holes(post_mask[0], structure=None, output=None, origin=0)
    return post_mask


def get_recist_from_crop_mask(crop_mask):
    if np.max(crop_mask) == 0:
        return None
    mcr = largest_connect_component(crop_mask)
    padded_binary_mask = np.pad(mcr, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    contour = contours[0]
    contour = close_contour(contour)
    contour = measure.approximate_polygon(contour, 2)
    contour = np.flip(contour, axis=1)
    long_diameter, short_diameter, long_diameter_len, short_diameter_len = find_recist_in_contour(contour)
    recist = np.vstack((contour[long_diameter], contour[short_diameter]))
    # recist = recist.reshape((-1))
    return recist


def prob2mask(mask, th=0.5):
    if isinstance(mask, torch.Tensor):
        binary_mask = mask.clone()
    else:
        binary_mask = mask.copy()
    binary_mask[binary_mask >= th] = 1
    binary_mask[binary_mask < th] = 0
    return binary_mask.astype(np.uint8)


def roi_mask2slice_mask(roi_mask, roi, slice_shape):
    mask = np.zeros(shape=slice_shape, dtype=np.uint8)
    # to int
    roi = roi.astype(np.int)
    roi_mask = cv2.resize(roi_mask, (roi[2], roi[3]), interpolation=cv2.INTER_NEAREST)
    # to xyxy
    xyxy_roi = roi.copy()
    xyxy_roi[2: 4] += xyxy_roi[0:2]
    mask[xyxy_roi[1]: xyxy_roi[3], xyxy_roi[0]:xyxy_roi[2]] = roi_mask
    return mask


def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array,
    1 - mask,
    0 - background

    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str = '', shape: tuple = (512, 512)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    total_elem_num = reduce((lambda x, y: x * y), list(shape))
    img = np.zeros(total_elem_num, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def bbox2gcmask(bbox, pts, img):
    bbox = bbox.squeeze().astype(np.int)
    mask = cv2.GC_BGD * np.ones(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    iterCount = 10
    pts = np.reshape(pts, (-1, 2))
    pts_convex = pts.copy()
    pts_convex[1: 3] = np.flip(pts_convex[1: 3], axis=0)
    cv2.rectangle(mask, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), cv2.GC_PR_FGD, -1)
    cv2.fillConvexPoly(mask, pts_convex.astype(np.int), cv2.GC_FGD)
    cv2.grabCut(img, mask, bbox, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_MASK)
    result = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result[result != 0] = 1
    return result


def triple2gcmask(triple, img):
    color_img = vis_img(img, None)
    may_fg_mask = triple.copy()
    may_fg_mask[may_fg_mask != 0] = 1
    may_fg_mask = binary_fill_holes(may_fg_mask, structure=None, output=None, origin=0)
    mask_instance, num = measure.label(may_fg_mask, connectivity=2, return_num=True)
    g_mask_list = []
    for i in range(1, num + 1):
        deal = mask_instance.copy()
        deal[deal != i] = 0
        deal[deal != 0] = 1
        cur_triple = deal * triple
        cur_triple = np.expand_dims(cur_triple, axis=-1).astype(np.uint8)
        bbox = mask2det(deal, 0, True, True)
        bbox = bbox.convert('xywh').get_array()[0]
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        iterCount = 10
        try:
            cv2.grabCut(color_img, cur_triple, tuple(bbox), bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_MASK)
            g_mask = np.where((cur_triple == 2) | (cur_triple == 0), 0, 1).astype('uint8')
        except:
            g_mask = np.zeros_like(mask_instance)
        g_mask_list.append(g_mask.squeeze())
    g_mask = mask_union(g_mask_list) if g_mask_list else np.zeros_like(mask_instance)
    return g_mask


def recistPts2diamond_mask(recist_pts_array, shape, value=1):
    recist_pts_array = np.array(recist_pts_array).reshape((-1, 4, 2))
    recist_pts_array = recist_pts_array.copy()
    recist_pts_array[:, 1: 3] = np.flip(recist_pts_array[:, 1: 3], axis=1)
    mask = np.zeros(shape, np.uint8)
    for recist_pts in recist_pts_array:
        cv2.fillPoly(mask, [recist_pts.astype(np.int)], value)
    return mask


def recistPts2circle_mask(recist_pts, shape, value=1):
    mask = np.zeros(shape, np.uint8)
    recist_pts_array = np.array(recist_pts).reshape((-1, 4, 2))
    for recist_pts in recist_pts_array:
        (x, y), radius = cv2.minEnclosingCircle(recist_pts.astype(np.int))
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(mask, center, radius, value, -1)
    return mask


def get_mask_center(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = (sum(contours[0]) / len(contours[0])).astype(np.int)[0]
    return cx, cy
