from functools import partial

import numpy as np
import cv2
import ast
from copy import deepcopy

from utils.geometry import euclidean_distance


def _convert_from_string(data):
    for char in '[]':
        data = data.replace(char, '')

    rows = data.split(';')
    newdata = []
    count = 0
    for row in rows:
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(ast.literal_eval, temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError("Rows not the same size.")
        count += 1
        newdata.append(newrow)
    return newdata


class BBoxes(np.ndarray):
    def __new__(cls, data, dtype=None, copy=True, mode='xyxy'):
        if isinstance(data, BBoxes):
            dtype2 = data.dtype
            if dtype is None:
                dtype = dtype2
            if (dtype2 == dtype) and (not copy):
                return data
            return data.astype(dtype)

        if isinstance(data, np.ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = np.dtype(dtype)
            new = data.view(cls)
            new.mode = mode
            if intype != data.dtype:
                return new.astype(intype)
            if copy:
                return new.copy()
            else:
                return new

        if isinstance(data, str):
            data = _convert_from_string(data)

        # now convert data to an array
        arr = np.array(data, dtype=dtype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape
        if ndim > 2:
            raise ValueError("bbox must be 2-dimensional")
        elif ndim == 0:
            shape = (1, 1)
        elif ndim == 1:
            shape = (1, shape[0])

        order = 'C'
        if (ndim == 2) and arr.flags.fortran:
            order = 'F'

        if not (order or arr.flags.contiguous):
            arr = arr.copy()

        ret = np.ndarray.__new__(cls, shape, arr.dtype,
                                 buffer=arr,
                                 order=order)
        ret.mode = mode
        return ret

    def __array_finalize__(self, obj):
        self._getitem = False
        self.mode = getattr(obj, 'mode', 'xyxy')
        if isinstance(obj, BBoxes) and obj._getitem: return
        ndim = self.ndim
        if ndim == 2:
            return
        if ndim > 2:
            newshape = tuple([x for x in self.shape if x > 1])
            ndim = len(newshape)
            if ndim == 2:
                self.shape = newshape
                return
            elif ndim > 2:
                raise ValueError("shape too large to be a BBoxes.")
        else:
            newshape = self.shape
        if ndim == 0:
            self.shape = (1, 1)
        elif ndim == 1:
            self.shape = (1, newshape[0])
        self.mode = getattr(obj, 'mode', 'xyxy')
        return

    def convert(self, mode):
        if self.mode == mode:
            return self
        self[:, 2:4] = self[:, 2:4] + {'xywh': -1, 'xyxy': 1}[mode] * self[:, 0:2]
        self.mode = mode
        return self

    def get_box(self, mode):
        box = deepcopy(self)
        return box.convert(mode)

    def get_array(self, mode=None):
        if mode is None:
            return self.__array__()
        temp = BBoxes(self).get_box(mode)
        return temp.get_array()

    def get_width(self):
        return self[:, 2] - {'xywh': 0, 'xyxy': 1}[self.mode] * self[:, 0]

    def get_height(self):
        return self[:, 3] - {'xywh': 0, 'xyxy': 1}[self.mode] * self[:, 1]


def bbox2bbox_channel(bbox, shape, dilate_iterations=0):
    bbox_channel = np.zeros(shape)
    bboxes_ = BBoxes(bbox)
    if bbox is None or bboxes_.shape[1] == 0:
        return bbox_channel
    bboxes_.convert('xyxy')
    for bbox in bboxes_.get_array():
        cv2.rectangle(bbox_channel, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 1, 1)
    if dilate_iterations != 0:
        bbox_channel = cv2.dilate(bbox_channel, (3, 3), dilate_iterations)
    return bbox_channel


def bbox2bbox_mask(bbox, shape):
    bbox_channel = np.zeros(shape)
    bboxes_ = BBoxes(bbox)
    if bbox is None or bboxes_.shape[1] == 0:
        return bbox_channel
    bboxes_.convert('xyxy')
    for bbox in bboxes_.get_array():
        cv2.rectangle(bbox_channel, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 1, -1)
    return bbox_channel


def bbox_transform(bbox, value, index, op):
    re_bbox = BBoxes(bbox)
    re_bbox.convert('xywh')
    bbox[:, index] = bbox[:, index] + op * value
    return re_bbox


def crop_img_roi(img, bbox, roi_shape=None, interpolation=None):
    assert len(bbox) == 1
    bbox = bbox.astype(np.int)
    bbox = bbox.get_array('xyxy').squeeze()
    img_roi = img[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    if roi_shape is not None:
        img_roi = cv2.resize(img_roi, roi_shape[::-1], interpolation=interpolation)
    return img_roi


bbox_up = partial(bbox_transform, index=1, op=-1)
bbox_left = partial(bbox_transform, index=0, op=-1)
bbox_down = partial(bbox_transform, index=1, op=1)
bbox_right = partial(bbox_transform, index=0, op=1)
bbox_horizontal_shrink = partial(bbox_transform, index=2, op=-1)
bbox_horizontal_expansion = partial(bbox_transform, index=2, op=1)
bbox_vertical_shrink = partial(bbox_transform, index=3, op=-1)
bbox_vertical_expansion = partial(bbox_transform, index=3, op=1)


def get_roi_from_bbox_and_recist(bbox_xyxy, recist_pts, w, h, roi_type='center_extend'):
    long_distance = euclidean_distance(recist_pts[0], recist_pts[1])
    bx1, by1, bx2, by2 = bbox_xyxy
    if roi_type == 'center_extend':
        cx, cy = (bx1 + bx2) / 2., (by1 + by2) / 2.
        edge1, edge2 = [[cx, cy] for _ in range(2)]
    else:
        edge1, edge2 = [bbox_xyxy[:2], bbox_xyxy[2:]]
    x1, y1 = tuple(map(lambda x: max(int(x - long_distance), 0), edge1))
    x2, y2 = tuple(map(lambda x: min(int(x[0] + long_distance), x[1]), zip(edge2, [w, h])))
    return x1, y1, x2, y2
