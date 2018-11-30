from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch


def box_form(temporal_dim, scale_ratios, feat_len):
    centers = np.array([], dtype=np.float32)
    widths = np.array([], dtype=np.float32)
    start_ids = np.array([], dtype=np.int)
    end_ids = np.array([], dtype=np.int)

    for i in range(len(temporal_dim)):
        num_anchor = temporal_dim[i]
        center = np.repeat(np.linspace(0, num_anchor - 1, num_anchor, dtype=np.float32), 3)
        center = (center + 0.5) / num_anchor
        width = np.tile(scale_ratios, num_anchor)
        width = width / num_anchor
        centers = np.concatenate((centers, center))
        widths = np.concatenate((widths, width))

        start_idx = np.repeat(np.linspace(0, num_anchor - 1, num_anchor, dtype=np.int), 3)
        start_idx = 2*(i+1)*start_idx-2*(i+1)+1
        start_idx[start_idx<0] = 0
        end_idx = np.repeat(np.linspace(0, num_anchor - 1, num_anchor, dtype=np.int), 3)
        end_idx = 2*(i+1)*end_idx+2*(i+1)-1
        end_idx[end_idx>feat_len] = feat_len
        start_ids = np.concatenate((start_ids, start_idx))
        end_ids = np.concatenate((end_ids, end_idx))

    starts = centers - widths / 2
    ends = centers + widths / 2

    centers = np.expand_dims(centers, 1)
    widths = np.expand_dims(widths, 1)
    starts = np.expand_dims(starts, 1)
    ends = np.expand_dims(ends, 1)
    temporal_box = np.concatenate((centers, widths, starts, ends), axis=1)

    start_ids = np.expand_dims(start_ids, 1)
    end_ids = np.expand_dims(end_ids, 1)
    temporal_feat_ids = np.concatenate((start_ids, end_ids), axis=1)
    return temporal_box, temporal_feat_ids


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


def match(pred_box, target_box, threshold, class_ids=None):
    t_iou = jaccard(pred_box, target_box)

    best_anchor_idx = np.argmax(t_iou, axis=0)
    best_target_iou = np.amax(t_iou, axis=1)
    best_target_idx = np.argmax(t_iou, axis=1)
    best_target_iou[best_anchor_idx] = 1.0
    for i in range(best_anchor_idx.shape[0]):
        best_target_idx[best_anchor_idx[i]] = i

    matches = target_box[best_target_idx]
    if class_ids is None:
        labels = best_target_idx + 1
        labels[best_target_iou < threshold] = 0
    else:
        labels = class_ids[best_target_idx]
        labels[best_target_iou < threshold] = 0
    events = labels.copy()
    events[labels > 0] = 1
    return matches, events, labels, best_target_iou


def get_pos_events(feats, anchors, temporal_dim, target_evn_box, target_sent,
                   target_cls_box, target_cls_ids, scale_ratios, threshold=0.7):
    # Tensor to numpy
    target_evn_box = target_evn_box.data.cpu().numpy()
    target_cls_box = target_cls_box.data.cpu().numpy()
    target_cls_ids = target_cls_ids.data.cpu().numpy()

    # Get temporal box
    t_box, t_feat_ids = box_form(temporal_dim, scale_ratios, feats.shape[2])

    # Get match info between temporal_boxes and ground-truth boxes
    gt_evn_box = target_evn_box[0,:,:4]
    gt_cls_box = target_cls_box[0,:,:4]
    evn_matches, evn_pos, evn_labels, evn_iou = match(t_box, gt_evn_box, threshold)
    cls_matches, cls_pos, cls_labels, cls_iou = match(t_box, gt_cls_box, threshold, class_ids=target_cls_ids[0])

    # Get positive proposals
    event_ids = np.array(np.where(evn_pos > 0)[0])
    num_events = event_ids.shape[0]
    pos_labels = torch.from_numpy(evn_labels[event_ids])
    pos_sents = target_sent[0, pos_labels-1, :]
    pos_anchors = anchors[0, :, event_ids].transpose(1, 0)
    pos_feats = list()
    for i in range(num_events):
        pos_feats.append(feats[0,:,t_feat_ids[event_ids[i],0]:t_feat_ids[event_ids[i],1]])

    return pos_feats, pos_anchors, pos_sents, t_box, \
           (evn_matches, evn_pos, evn_labels, evn_iou), \
           (cls_matches, cls_pos, cls_labels, cls_iou)


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


def sample_proposal(feats, predictions, alpha1, alpha2, n_feats,
                    scale_ratios=np.asarray([1, 1.25, 1.5], dtype=np.float32),
                    threshold=0.7, duration=0.0):

    def non_max_suppression_fast(boxes, overlapThresh, idxs):
        ### Ref. https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 2]
        x2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1)
        # idxs = np.argsort(x1)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1)

            # compute the ratio of overlap
            overlap = w / (area[idxs[:last]] + area[i] - w)

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return pick

    loc, evt, cls, _, temporal_dim = predictions

    # Get temporal box
    t_box, _ = box_form(temporal_dim, scale_ratios, feats.shape[2])
    num_feats = feats.shape[2]
    num_anchors = t_box.shape[0]
    feats_center = np.linspace(0, num_feats - 1, num_feats, dtype=np.float32)
    feats_center = (2 * feats_center + 1) / (2.0 * num_feats)

    # Get positive samples
    cand_score = evt.view(-1, 2)[:, 1]
    sort_score = cand_score.sort(0, True)[0].cpu().numpy()
    sort_score_idx = cand_score.sort(0)[1].cpu().numpy()
    t = non_max_suppression_fast(t_box, 0.5, sort_score_idx)
    best_score = torch.max(cand_score).item()
    if best_score < threshold:
        score = 0
        pos_idx = torch.zeros(cand_score.shape, dtype=torch.uint8)
        i = 0
        threshold = threshold + duration/1000.0
        while score < threshold and i < len(t):
            pos_idx[t[i]] = 1
            score = score + cand_score[t[i]]
            i = i + 1
    else:
        pos_idx = cand_score >= threshold
    predictions_center = t_box[:, 0] + alpha1 * t_box[:, 1] * loc[0].data[:, 0]
    predictions_width = t_box[:, 1] * torch.exp(alpha2 * loc[0].data[:, 1])
    pos_center = predictions_center[pos_idx]
    pos_width = predictions_width[pos_idx]
    pos_center = pos_center.data.cpu().numpy()
    pos_width = pos_width.data.cpu().numpy()

    # for Torch
    dim_feats = feats.shape[1]
    num_events = pos_center.shape[0]
    pos_feats = torch.zeros(num_events, dim_feats, n_feats)
    time_stamp = np.zeros((num_events, 2))
    for i in range(num_events):
        if pos_width[i] > 1.0:
            pos_width[i] = 1.0
        if pos_center[i] - pos_width[i] / 2.0 < 0.0:
            pos_center[i] = pos_width[i] / 2.0
        elif pos_center[i] + pos_width[i] / 2.0 > 1.0:
            pos_center[i] = 1 - pos_width[i] / 2.0
        start_time = pos_center[i]-pos_width[i]/2.0
        end_time = pos_center[i]+pos_width[i]/2.0
        if start_time < 0.0 or end_time > 1.0 or start_time > end_time:
            raise Exception("time : out of range")
        start_idx = np.sum(start_time > feats_center)
        end_idx = np.sum(end_time >= feats_center)
        if start_idx == end_idx:
            raise Exception("time length is 0")
        x = feats[0, :, start_idx:end_idx]
        x = x.data.cpu().numpy()
        l = x.shape[1]
        choice_idx = np.round(np.linspace(0, l - 1, n_feats, endpoint=True)).astype(int)
        f = x.take(choice_idx, axis=1)
        pos_feats[i, :, :] = torch.from_numpy(f).float()
        time_stamp[i, 0] = start_time
        time_stamp[i, 1] = end_time

    return pos_feats, time_stamp
