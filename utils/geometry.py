import re

import cv2
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import math


def intersection(start1, end1, start2, end2):
    x1, y1, x2, y2, x3, y3, x4, y4 = *start1, *end1, *start2, *end2
    d = np.linalg.det(np.array([[x1 - x2, x4 - x3], [y1 - y2, y4 - y3]]))
    p = np.linalg.det(np.array([[x4 - x2, x4 - x3], [y4 - y2, y4 - y3]]))
    q = np.linalg.det(np.array([[x1 - x2, x4 - x2], [y1 - y2, y4 - y2]]))
    if d != 0:
        lam, eta = p / d, q / d
        if not (0 <= lam <= 1 and 0 <= eta <= 1):
            return None, None
        return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2
    return None, None


def recist2det(recist_pts_array, pad_size, height, width):
    """
        recist_pts_array: (N, 4, 2)
    """
    recist_pts_array = np.array(recist_pts_array).reshape((-1, 4, 2))
    x1 = np.maximum(np.min(recist_pts_array[:, :, 0], axis=1, keepdims=True) - pad_size, 0)
    x2 = np.minimum(np.max(recist_pts_array[:, :, 0], axis=1, keepdims=True) + pad_size, width)
    y2 = np.minimum(np.max(recist_pts_array[:, :, 1], axis=1, keepdims=True) + pad_size, height)
    y1 = np.maximum(np.min(recist_pts_array[:, :, 1], axis=1, keepdims=True) - pad_size, 0)
    dets = np.concatenate((x1, y1, x2, y2), axis=1)
    return dets


def vector_product(vec1, vec2):
    vec1, vec2 = list(map(np.array, [vec1, vec2]))
    return np.sum(vec1 * vec2)


def vector_norm(vec, order=2):
    vec = np.array(vec)
    return np.linalg.norm(vec, ord=order, axis=None, keepdims=False)


def get_cos(vec1, vec2):
    return abs(vector_product(vec1, vec2) / vector_norm(vec1) / vector_norm(vec2))


def euclidean_distance(pt1, pt2):
    return math.sqrt((1 * (pt1[0] - pt2[0])) ** 2 + (1 * (pt1[1] - pt2[1])) ** 2)


def recist_euclidean_distance(pt1, pt2, spacing):
    return math.sqrt((spacing[0] * (pt1[0] - pt2[0])) ** 2 + (spacing[1] * (pt1[1] - pt2[1])) ** 2)
    # return math.sqrt((1 * (pt1[0] - pt2[0])) ** 2 + (1 * (pt1[1] - pt2[1])) ** 2)


def rotate_img(pts, center, ori_img, clockwise=True):
    if pts[0][0] == pts[1][0]:
        return ori_img, 0
    k = (pts[0][1] - pts[1][1]) / (pts[0][0] - pts[1][0])
    rad = math.atan(k)
    degree = 180 * rad / math.pi
    if not clockwise:
        degree = -degree
    rmat = cv2.getRotationMatrix2D(center, degree, 1)
    r_img = cv2.warpAffine(ori_img, rmat, ori_img.shape[:2])
    return r_img, degree


def rotate_img_tuple(pts, center, ori_img_tuple):
    if pts[0][1] == pts[1][1]:
        return ori_img_tuple
    k = (pts[0][1] - pts[1][1]) / (pts[0][0] - pts[1][0])
    rad = math.atan(k)
    degree = 180 * rad / math.pi
    rmat = cv2.getRotationMatrix2D(center, degree, 1)
    re = []
    for img in ori_img_tuple:
        re.append(cv2.warpAffine(img, rmat, img.shape[:2]))
    return re


def find_recist_in_contour(contour):
    if len(contour) == 1:
        return np.array([0, 0]), np.array([0, 0]), 1, 1
    D = squareform(pdist(contour)).astype('float32')
    long_diameter_len = D.max()
    endpt_idx = np.where(D == long_diameter_len)
    endpt_idx = np.array([endpt_idx[0][0], endpt_idx[1][0]])
    long_diameter_vec = (contour[endpt_idx[0]] - contour[endpt_idx[1]])[:, None]

    side1idxs = np.arange(endpt_idx.min(), endpt_idx.max() + 1)
    side2idxs = np.hstack((np.arange(endpt_idx.min() + 1), np.arange(endpt_idx.max(), len(contour))))
    perp_diameter_lens = np.empty((len(side1idxs),), dtype=float)
    perp_diameter_idx2s = np.empty((len(side1idxs),), dtype=int)
    for i, idx1 in enumerate(side1idxs):
        short_diameter_vecs = contour[side2idxs] - contour[idx1]
        dot_prods_abs = np.abs(np.matmul(short_diameter_vecs, long_diameter_vec))
        idx2 = np.where(dot_prods_abs == dot_prods_abs.min())[0]
        if len(idx2) > 1:
            idx2 = idx2[np.sum(short_diameter_vecs[idx2] ** 2, axis=1).argmax()]
        idx2 = side2idxs[idx2]  # find the one that is perpendicular with long axis
        perp_diameter_idx2s[i] = idx2
        perp_diameter_lens[i] = D[idx1, idx2]
    short_diameter_len = perp_diameter_lens.max()
    short_diameter_idx1 = side1idxs[perp_diameter_lens.argmax()]
    short_diameter_idx2 = perp_diameter_idx2s[perp_diameter_lens.argmax()]
    short_diameter = np.array([short_diameter_idx1, short_diameter_idx2])
    long_diameter = endpt_idx

    return long_diameter, short_diameter, long_diameter_len, short_diameter_len


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def get_cross_point(recist_point, height=512):
    re = np.zeros((4, 2))
    x = recist_point[::2]
    y = height - recist_point[1::2]
    mid_x, mid_y = np.mean(x.astype(np.float)), np.mean(y.astype(np.float))
    region_map = {(1, 1): 0, (1, -1): 1, (-1, -1): 2, (-1, 1): 3}
    judge = lambda x1, y1: (1 if y1 - x1 + mid_x - mid_y >= 0 else -1, 1 if y1 + x1 - mid_x - mid_y >= 0 else -1)
    long1_r = region_map[judge(x[0], y[0])]
    if long1_r == 0 or long1_r == 2:
        re[long1_r] = np.array([x[0], height - y[0]])
        re[2 - long1_r] = np.array([x[1], height - y[1]])
        short1_r = 1 if x[2] <= x[3] else 3
        re[short1_r] = np.array([x[2], height - y[2]])
        re[4 - short1_r] = np.array([x[3], height - y[3]])
    else:
        re[long1_r] = np.array([x[0], height - y[0]])
        re[4 - long1_r] = np.array([x[1], height - y[1]])
        short1_r = 2 if y[2] <= y[3] else 0
        re[short1_r] = np.array([x[2], height - y[2]])
        re[2 - short1_r] = np.array([x[3], height - y[3]])
    return re


def transpose_by_cosine_matrix(input_array, target_cosine_matrix=None, input_meta=None, ori_cosine_matrix=None):
    """
    Not an ambitious implementation: Cosine matrix only support permutation matrix.
    """
    nature_permuation_matrix = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape((3, 3))
    target_cosine_matrix = nature_permuation_matrix if target_cosine_matrix is None \
        else np.array(target_cosine_matrix).reshape(3, 3)
    if ori_cosine_matrix is not None:
        # this case we first transpose it to nature coordinates system
        target = np.linalg.inv(np.round(ori_cosine_matrix))
        input_array, input_meta = transpose_by_cosine_matrix(input_array, target_cosine_matrix=target,
                                                             input_meta=input_meta)
    # Now the input array and meta is in nature coordinates system
    if target_cosine_matrix is None and ori_cosine_matrix is None:
        # if target_cosine_matrix and ori_cosine_matrix all nature coordinates system
        return input_array, input_meta
    transpose = np.argmax(abs(target_cosine_matrix), axis=0)
    flip = np.sum(target_cosine_matrix, axis=0)
    # Apply transformation to image volume
    output_array = np.transpose(input_array, tuple(transpose))
    output_array = output_array[tuple(slice(None, None, int(f)) for f in flip)]
    if input_meta is None:
        return output_array, None
    output_meta = {re.sub(r'ori_', 'target_', key): np.array([value[t] for t in transpose])
                   for key, value in input_meta.items()
                   if key != 'ori_direction'}
    output_meta['target_direction'] = target_cosine_matrix
    return output_array, output_meta


def crop_points_to_roi(points_list, x1, y1):
    if not isinstance(points_list, list):
        points_list = [points_list]
    croped_points_list = []
    for points in points_list:
        points = np.array(points)
        shape = points.shape
        points = np.reshape(points, (-1, 2))
        points[..., 0] -= x1
        points[..., 1] -= y1
        croped_points_list.append(points.reshape(shape))
    return croped_points_list


def cal_iou_from_polygon(polygon1, polygon2):
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou

