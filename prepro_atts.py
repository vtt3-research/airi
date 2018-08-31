from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import h5py
import argparse
import numpy as np
from collections import Counter


def _build_att_class(atts):
    counter = Counter()
    for v_name in atts['database']:
        v_data = atts['database'][v_name]
        if v_data['subset'] == 'training':
            for att in v_data['annotations']:
                counter[att['label']] += 1

    labels = [label for label in counter]
    label_to_idx = {}
    idx = 0
    for label in labels:
        label_to_idx[label] = idx
        idx += 1

    return label_to_idx


def _build_features(atts, feats, label_to_idx, flag='training'):
    # flag : "training" or "validation"
    att_feats = list()
    att_labels = list()

    count = 0
    for v_name in atts['database']:
        v_data = atts['database'][v_name]
        if v_data['subset'] == flag:
            v_feats = feats[str('v_') + v_name]['c3d_features'].value
            duration = v_data['duration']
            for att in v_data['annotations']:
                start, end = att['segment']
                start = v_feats.shape[0]*start/duration
                end = v_feats.shape[0]*end/duration
                start = np.round(start).astype(int)
                end = np.round(end).astype(int)
                if start == end:
                    if end != v_feats.shape[0]:
                        end = end+1
                    else:
                        start = start-1
                att_feat = v_feats[start:end]
                att_feat = np.sum(att_feat, axis=0)/float(end-start)
                att_label = label_to_idx[att['label']]
                att_feats.append(att_feat)
                att_labels.append(att_label)

            count += 1
            if count % 1000 == 0:
                print("{} : Processed {} ...".format(flag, count))

    print("Finished {} dataset".format(flag))
    att_feats = np.asarray(att_feats)
    att_labels = np.asarray(att_labels)
    return att_feats, att_labels


def main(params):
    atts = json.load(open(params.input_atts, 'r'))
    feats = h5py.File(params.input_feats, 'r')
    label_to_idx = _build_att_class(atts)
    train_att_feats, train_att_labels = _build_features(atts, feats, label_to_idx, 'training')
    val_att_feats, val_att_labels = _build_features(atts, feats, label_to_idx, 'validation')

    json.dump(label_to_idx, open(params.output_labeltoidx, 'w'))
    f_lb = h5py.File(params.output_train, "w")
    f_lb.create_dataset('features', data=train_att_feats)
    f_lb.create_dataset('labels', data=train_att_labels)
    f_lb.close()

    f_lb = h5py.File(params.output_val, "w")
    f_lb.create_dataset('features', data=val_att_feats)
    f_lb.create_dataset('labels', data=val_att_labels)
    f_lb.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_atts', default='./data/activity_net.v1-3.min.json',
                        help='input ActivityNet200 dataset (json file)')
    parser.add_argument('--input_feats', default='./data/actnet/sub_activitynet_v1-3.c3d.hdf5',
                        help='input ActivityNet200 c3d feature file (hdf5 file)')
    parser.add_argument('--output_train', default='./data/actnet/att_train.hdf5',
                        help='output file with features and labels for training (hdf5 file)')
    parser.add_argument('--output_val', default='./data/actnet/att_val.hdf5',
                        help='output file with features and labels for validation (hdf5 file)')
    parser.add_argument('--output_labeltoidx', default='./data/actnet/att_label_to_idx.json',
                        help='label-to-idx file (json file)')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
