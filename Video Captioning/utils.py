from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch


def box_form(temporal_dim, scale_ratios):
    centers = np.array([], dtype=np.float32)
    widths = np.array([], dtype=np.float32)

    for i in range(len(temporal_dim)):
        num_anchor = temporal_dim[i]
        center = np.repeat(np.linspace(0, num_anchor - 1, num_anchor, dtype=np.float32), 3)
        center = (center + 0.5) / num_anchor
        width = np.tile(scale_ratios, num_anchor)
        width = width / num_anchor
        centers = np.concatenate((centers, center))
        widths = np.concatenate((widths, width))
    starts = centers - widths / 2
    ends = centers + widths / 2

    centers = np.expand_dims(centers, 1)
    widths = np.expand_dims(widths, 1)
    starts = np.expand_dims(starts, 1)
    ends = np.expand_dims(ends, 1)
    temporal_box = np.concatenate((centers, widths, starts, ends), axis=1)
    return temporal_box


def intersection(a, b):
    start_a = np.tile(np.expand_dims(a[:, 2], 1), b.shape[0])
    end_a = np.tile(np.expand_dims(a[:, 3], 1), b.shape[0])
    start_b = np.repeat(np.expand_dims(b[:, 2], 0), a.shape[0], axis=0)
    end_b = np.repeat(np.expand_dims(b[:, 3], 0), a.shape[0], axis=0)
    max_start = np.maximum(start_a, start_b)
    min_end = np.minimum(end_a, end_b)
    inter = np.ndarray.clip((min_end - max_start), 0.0, 1.0)
    return inter


def jaccard(a, b):
    inter = intersection(a, b)
    width_a = np.tile(np.expand_dims(a[:, 1], 1), b.shape[0])
    width_b = np.repeat(np.expand_dims(b[:, 1], 0), a.shape[0], axis=0)
    union = width_a + width_b - inter
    return inter / union


def match(pred_box, target_box, threshold):
    t_iou = jaccard(pred_box, target_box)

    best_anchor_idx = np.argmax(t_iou, axis=0)
    best_target_iou = np.amax(t_iou, axis=1)
    best_target_idx = np.argmax(t_iou, axis=1)
    best_target_iou[best_anchor_idx] = 1.0
    for i in range(best_anchor_idx.shape[0]):
        best_target_idx[best_anchor_idx[i]] = i

    matches = target_box[best_target_idx]
    labels = best_target_idx + 1
    labels[best_target_iou < threshold] = 0
    events = labels.copy()
    events[labels > 0] = 1
    return matches, events, labels


def clip_des_score(feats, des, temporal_dim, targets, sents,
                   scale_ratios=np.asarray([1, 1.25, 1.5], dtype=np.float32),
                   threshold=0.7, use_gpu=True):
    # Tensor to numpy
    targets = targets.data.cpu().numpy()

    # Get temporal box
    t_box = box_form(temporal_dim, scale_ratios)

    # Calc clip-level descriptiveness score
    num_feats = feats.shape[2]
    num_anchors = t_box.shape[0]
    feats_center = np.linspace(0, num_feats - 1, num_feats, dtype=np.float32)
    feats_center = (2 * feats_center + 1) / (2.0 * num_feats)

    # for Torch
    x = des[:, :, 0].expand(num_feats, -1)
    y = torch.zeros((num_feats, num_anchors))
    if use_gpu:
        y = y.cuda()
    t_box_start = np.repeat(np.expand_dims(t_box[:, 2], 0), num_feats, axis=0)
    t_box_end = np.repeat(np.expand_dims(t_box[:, 3], 0), num_feats, axis=0)
    feats_center_tile = np.tile(np.expand_dims(feats_center, 1), num_anchors)
    idx = np.logical_and(t_box_start <= feats_center_tile, feats_center_tile <= t_box_end)
    idx = torch.from_numpy(idx.astype(int))
    if use_gpu:
        idx = idx.cuda()
    y[idx == 1] = x[idx == 1]
    clip_des_s = torch.sum(y, 1)
    clip_des_s = clip_des_s/torch.sum(idx, 1).float()

    # Get match info between temporal_boxes and ground-truth boxes
    g_box = targets[0, :, :4]
    matches, events, labels = match(t_box, g_box, threshold)

    # for Torch
    dim_feats = feats.shape[1]
    event_ids = np.array(np.where(events > 0)[0])
    num_events = event_ids.shape[0]
    pos_labels = torch.from_numpy(labels[event_ids])
    pos_sents = sents[0, pos_labels-1, :]
    # pos_labels = torch.unsqueeze(pos_labels, dim=1)
    pos_feats = torch.zeros(num_events, dim_feats)
    for i in range(num_events):
        start_idx = np.sum(t_box[event_ids[i], 2] > feats_center)
        end_idx = np.sum(t_box[event_ids[i], 3] >= feats_center)
        if start_idx == end_idx:
            print("time length is 0")
        x = clip_des_s[start_idx:end_idx]/torch.sum(clip_des_s[start_idx:end_idx])
        y = feats[0, :, start_idx:end_idx]
        pos_feats[i] = torch.sum(x*y, 1)

    return pos_feats, pos_sents, t_box, matches, events, event_ids


def match_gt(pred_box, target_box):
    t_iou = jaccard(pred_box, target_box)
    best_anchor_idx = np.argmax(t_iou, axis=0)
    return best_anchor_idx


def get_gt_proposal(feats, des, temporal_dim, targets,
                    scale_ratios=np.asarray([1, 1.25, 1.5], dtype=np.float32),
                    use_gpu=True):
    if targets.size(0) > 1:
        raise ("batch size must be set as 1")

    # Tensor to numpy
    targets = targets.data.cpu().numpy()

    # Get temporal box
    t_box = box_form(temporal_dim, scale_ratios)

    # Calc clip-level descriptiveness score
    num_feats = feats.shape[2]
    num_anchors = t_box.shape[0]
    feats_center = np.linspace(0, num_feats - 1, num_feats, dtype=np.float32)
    feats_center = (2 * feats_center + 1) / (2.0 * num_feats)

    # for Torch
    x = des[:, :, 0].expand(num_feats, -1)
    y = torch.zeros((num_feats, num_anchors))
    if use_gpu:
        y = y.cuda()
    t_box_start = np.repeat(np.expand_dims(t_box[:, 2], 0), num_feats, axis=0)
    t_box_end = np.repeat(np.expand_dims(t_box[:, 3], 0), num_feats, axis=0)
    feats_center_tile = np.tile(np.expand_dims(feats_center, 1), num_anchors)
    idx = np.logical_and(t_box_start <= feats_center_tile, feats_center_tile <= t_box_end)
    idx = torch.from_numpy(idx.astype(int))
    if use_gpu:
        idx = idx.cuda()
    y[idx == 1] = x[idx == 1]
    clip_des_s = torch.sum(y, 1)
    clip_des_s = clip_des_s/torch.sum(idx, 1).float()

    g_box = targets[0, :, :4]
    dim_feats = feats.shape[1]
    num_events = g_box.shape[0]
    pos_feats = torch.zeros(num_events, dim_feats)
    for i in range(num_events):
        start_idx = np.sum(g_box[i, 2] > feats_center)
        end_idx = np.sum(g_box[i, 3] >= feats_center)
        if start_idx == end_idx:
            if end_idx != feats.shape[2]:
                end_idx = end_idx + 1
            else:
                start_idx = start_idx - 1
        x = clip_des_s[start_idx:end_idx]/torch.sum(clip_des_s[start_idx:end_idx])
        y = feats[0, :, start_idx:end_idx]
        pos_feats[i] = torch.sum(x*y, 1)

    return pos_feats


def arr_to_word(arr, idx_to_word):
    out = ''
    for i in range(len(arr)):
        if arr[i] == 1:  # start token
            pass
        elif arr[i] == 2:  # end token
            out += '.'
            break
        elif arr[i] == 0:  # null token
            break
        out += idx_to_word[arr[i]] + ' '
    return out.strip()


def idx_to_sent(gen_result, idx_to_word):
    gen_result = gen_result.data.cpu().numpy()

    gen_sents = list()
    num_gen = gen_result.shape[0]
    for i in range(num_gen):
        gen_sents.append([arr_to_word(gen_result[i], idx_to_word)])

    return gen_sents
