import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math

########################
# debug
########################
from utils.tensor_operator import sigmoid_with_clamp


def get_feature(feature):
    return np.squeeze(feature.cpu().detach().numpy())


########################
# decode
########################

def topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind.long())
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def nms_with_bbox(detections, nms_threshold):
    # detections: [batch_size, 100, 5]
    post_detections = []
    post_pts = []
    scores_list = []
    for i in range(len(detections)):
        dets = detections[i]
        # reject detections with negative scores
        keep_inds = (dets[:, 4] > 0)
        dets = dets[keep_inds]
        boxes = dets[:, :4]
        scores = dets[:, 4]
        extreme_points = dets[:, 5:13]
        keep_inds = torchvision.ops.nms(boxes, scores, nms_threshold)
        boxes = boxes[keep_inds]
        scores = scores[keep_inds]
        scores_list.append(scores)
        extreme_points = extreme_points[keep_inds]
        scores = torch.unsqueeze(scores, dim=1)
        post_detections.append(boxes)
        post_pts.append(extreme_points)
    return post_detections, post_pts, scores_list


def bbox_match(gt_bboxes, pred_bboxes):
    # todo need implementation
    return pred_bboxes


def get_ij(height, width):
    re_i, re_j = torch.meshgrid([torch.arange(height), torch.arange(width)])
    return re_i.float().cuda(), re_j.float().cuda()


def sqrt(matrix):
    s = torch.ones_like(matrix)
    s[matrix < 0] = -1
    re = torch.sqrt(torch.abs(matrix))
    re = re.mul(s)
    return re


def draw_circle(center_y, center_x, radius2, i, j, height=128, width=128, activation=F.sigmoid):
    re = torch.zeros(size=(height, width), dtype=torch.float).cuda()
    c_y, c_x = center_y * torch.ones_like(re), center_x * torch.ones_like(re)
    radius = torch.sqrt(radius2)
    re = (radius - torch.sqrt(torch.pow(j - c_x, 2) + torch.pow(i - c_y, 2))) / (radius + 1e-7)
    re = re if activation is None else activation(re)
    return torch.unsqueeze(re, dim=0)


def draw_weight_circle(center_y, center_x, radius2, i, j, scale, height=128, width=128):
    re = torch.zeros(size=(height, width), dtype=torch.float).cuda()
    c_y, c_x = center_y * torch.ones_like(re), center_x * torch.ones_like(re)
    radius = torch.sqrt(radius2)
    re = scale * F.relu(radius - torch.sqrt(torch.pow(j - c_x, 2) + torch.pow(i - c_y, 2))) / (radius + 1e-7)
    return torch.unsqueeze(re, dim=0)


def get_top_yx(hm, height, width):
    hm = torch.reshape(hm, (-1,))
    _, idx = torch.topk(hm, 1)
    idx = idx[0]
    return idx // width, idx % width


def get_feature_and_pts_from_hm(edge_point_hm, center_hm, matrix_i, matrix_j):
    assert len(edge_point_hm) == 4
    c, height, width = center_hm.size()
    center_y, center_x = get_top_yx(center_hm, height, width)
    feature = torch.zeros_like(center_hm)
    pts = []
    half_pts = []
    for point_hm in edge_point_hm:
        pt_y, pt_x = get_top_yx(point_hm, height, width)
        cy, cx = (center_y + pt_y) / 2, (center_x + pt_x) / 2
        radius2 = (pt_y - cy) ** 2 + (pt_x - cx) ** 2
        feature += draw_circle(cy, cx, radius2, matrix_i, matrix_j, height, width)
        pts += [pt_y, pt_x]
        half_pts.append([cy, cx])
    return feature, pts, [center_y, center_x], half_pts


def get_weight_map(pts, center, matrix_i, matrix_j, height=128, width=128):
    pts_x, pts_y = pts[0::2], pts[1::2]
    ini_dis = torch.pow(pts_y[0] - center[0], 2) + torch.pow(pts_x[1] - center[1], 2)
    long2, short2 = ini_dis, ini_dis
    for i in range(1, 4):
        dis2 = torch.pow(pts_y[i] - center[0], 2) + torch.pow(pts_x[0] - center[1], 2)
        if dis2 > long2:
            long2 = dis2
        if dis2 < short2:
            short2 = dis2
    long_weight_map = draw_weight_circle(center[0], center[1], long2.float(), matrix_i, matrix_j, 0.5, height, width)
    short_weight_map = draw_weight_circle(center[0], center[1], short2.float(), matrix_i, matrix_j, 1, height, width)
    weight_map = torch.clamp(long_weight_map + short_weight_map, min=0, max=1)
    return weight_map


def get_batch_roi_feature_and_weight_map(batch_max_roi_num, roi_edge_hm, roi_center_hm, matrix_i, matrix_j):
    re = torch.zeros_like(roi_center_hm)
    weight_map = torch.zeros_like(roi_center_hm)
    batch_hf_pts = []
    batch_centers = []
    for i in range(0, batch_max_roi_num):
        edge_pt_hm = [roi_edge_hm[j][i] for j in range(len(roi_edge_hm))]
        center_hm = roi_center_hm[i]
        # kid_A = torch.cat(edge_pt_hm + [center_hm], dim=0)
        # kid_A, _ = torch.max(kid_A, dim=0, keepdim=True)
        # kid_A = (kid_A - torch.min(kid_A)) / (torch.max(kid_A) - torch.min(kid_A))
        # kid_A = torch.pow(kid_A, 10)
        # kid_A = (kid_A - torch.min(kid_A)) / (torch.max(kid_A) - torch.min(kid_A))
        feature, pts, center, hf_pts = get_feature_and_pts_from_hm(edge_pt_hm, center_hm, matrix_i, matrix_j)
        batch_hf_pts.append(hf_pts)
        batch_centers.append(center)
        roi_feature = feature
        roi_feature = (roi_feature - torch.min(roi_feature)) / (torch.max(roi_feature) - torch.min(roi_feature))
        re[i] = roi_feature
    return re, weight_map, batch_hf_pts, batch_centers


def get_roi_hm(t_heat, l_heat, b_heat, r_heat, ct_heat, boxes, roi_output_size):
    edge_hm_list = [t_heat, l_heat, b_heat, r_heat]
    edge_hm_list = [torchvision.ops.roi_align(edge_hm_list[i], boxes, output_size=roi_output_size) for i in range(4)]
    ct_hm = torchvision.ops.roi_align(ct_heat, boxes, output_size=roi_output_size)
    return edge_hm_list, ct_hm


def hm2feature(t_heat, l_heat, b_heat, r_heat, ct_heat, boxes, roi_output_size, matrix_i, matrix_j):
    edge_hm, ct_hm = get_roi_hm(t_heat, l_heat, b_heat, r_heat, ct_heat, boxes, roi_output_size)
    re, weight_map, batch_hf_pts, batch_ct_pts = get_batch_roi_feature_and_weight_map(len(boxes), edge_hm, ct_hm,
                                                                                      matrix_i, matrix_j)
    return re, weight_map, batch_hf_pts, batch_ct_pts


def reconstruction(segs, atts, weight_maps, batch_pts, batch_cts, matrix_i, matrix_j):
    re = torch.zeros_like(segs)
    for i in range(segs.size()[0]):
        seg, att, pts, ct, wp = segs[i, 0], atts[i, 0], batch_pts[i], batch_cts[i], weight_maps[i]
        re[i, 0] = election(seg, att, pts, ct, matrix_i, matrix_j)
    return re


def election(seg, att, pts, ct, matrix_i, matrix_j):
    if seg[ct] < 0.5:
        return att
    weight = torch.zeros_like(seg)
    for i in range(len(pts)):
        if seg[torch.round(pts[i][0]).int(), torch.round(pts[i][1]).int()] < 0.5:
            radius2 = torch.pow(pts[i][0] - ct[0], 2) + torch.pow(pts[i][1] - ct[1], 2)
            weight += torch.squeeze(draw_circle(pts[i][0], pts[i][1], radius2, matrix_i, matrix_j, activation=F.relu))
    weight = torch.clamp(weight, min=0, max=1)
    return seg * (1 - weight) + att * weight


def hw_norm(feature):
    re = torch.zeros_like(feature)
    for i in range(feature.size()[0]):
        hw = feature[i, 0]
        hw = (hw - torch.min(hw)) / (torch.max(hw) - torch.min(hw))
        re[i, 0] = hw
    return re


########################
# Geometric
########################
def intersection(start1, end1, start2, end2, th=1e-2):
    x1, y1, x2, y2, x3, y3, x4, y4 = *start1, *end1, *start2, *end2
    det = lambda a, b, c, d: a.mul(d) - b.mul(c)
    d = det(x1 - x2, x4 - x3, y1 - y2, y4 - y3)
    p = det(x4 - x2, x4 - x3, y4 - y2, y4 - y3)
    q = det(x1 - x2, x4 - x2, y1 - y2, y4 - y2)
    _d = d.detach().clone()
    d[d <= th] = th
    lam, eta = p.div(d), q.div(d)
    valid = (0 <= lam) * (lam <= 1) * (0 <= eta) * (eta <= 1) * (_d > th)
    return valid * (lam.mul(x1) + (1 - lam).mul(x2)), valid * (lam.mul(y1) + (1 - lam).mul(y2))


def inter_degree_score(start1, end1, start2, end2):
    x1, y1, x2, y2, x3, y3, x4, y4 = *start1, *end1, *start2, *end2
    v1, v2 = (x2 - x1, y2 - y1), (x4 - x3, y4 - y3)
    norm1, norm2 = torch.sqrt(v1[0] * v1[0] + v1[1] * v1[1]), torch.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    cos = abs((v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2 + 1e-2))
    degree_scroe = 1 - cos
    return degree_scroe


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0].mul(b[1]) - a[1].mul(b[0])

    div = det(xdiff, ydiff)
    div[div == 0] = 1e-5

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def sigmoid(x):
    x = sigmoid_with_clamp(x)
    return x


def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


def global_mean_pool_with_mask(in_tensor, mask):
    mask = mask.int()
    div = mask.sum(dim=-1, keepdims=True)
    mask = mask.unsqueeze(-1)
    mask = mask.repeat((1,) * len(mask.size()[:-1] )+ (in_tensor.size()[-1], ))
    in_tensor = in_tensor * mask
    mean_tensor = in_tensor.sum(dim=1) / div
    return mean_tensor
