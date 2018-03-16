from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import h5py
import numpy as np
import random


class LoadData(object):
    def __init__(self, input_json, input_h5, feature_path, batch_img, seq_per_img, train_only):
        """
        Args:
            input_json: label dataset file.
            input_h5: caption dataset file.
            feature_path: feature data path.
            batch_img: number of images per each batch.
            seq_per_img: number of captions per each images.
            train_only: seperate training set. if is 0, then training set include rest validation set.
        """

        self.batch_img = batch_img
        self.seq_per_img = seq_per_img
        self.batch_size = batch_img * seq_per_img

        # path of feature
        self.feature_path = feature_path

        # load json file
        self.info = json.load(open(input_json))
        self.word_to_idx = self.info['word_to_idx']
        self.voca_size = len(self.word_to_idx)

        # load h5 file
        self.h5_label_file = h5py.File(input_h5, 'r', driver='core')
        self.seq_length = self.h5_label_file['labels'].shape[1]
        self.label_start_idx = self.h5_label_file['label_start_idx'][:]
        self.label_end_idx = self.h5_label_file['label_end_idx'][:]
        self.num_images = self.label_start_idx.shape[0]

        # seperate train, val, test file
        self.split_idx = {'train': [], 'val': [], 'test': []}
        for idx in range(len(self.info['images'])):
            img = self.info['images'][idx]
            if img['split'] == 'train':
                self.split_idx['train'].append(idx)
            elif img['split'] == 'val':
                self.split_idx['val'].append(idx)
            elif img['split'] == 'test':
                self.split_idx['test'].append(idx)
            elif train_only == 0:  # restval
                self.split_idx['train'].append(idx)
        self.num_train_images = len(self.split_idx['train'])
        self.num_val_images = len(self.split_idx['val'])
        self.num_test_images = len(self.split_idx['test'])
        print('assigned %d images to split train' % self.num_train_images)
        print('assigned %d images to split val' % self.num_val_images)
        print('assigned %d images to split test' % self.num_test_images)

        # batch index for training
        self.batch_idx = np.arange(self.num_train_images)

    def _load_feature(self, file_name):
        feature_file = os.path.join(self.feature_path, file_name)
        with np.load(feature_file) as data:
            feature = data['feat']
        return feature

    def _load_boxes(self, file_name):
        feature_file = os.path.join(self.feature_path, file_name)
        with np.load(feature_file) as data:
            boxes = data['boxes']
        return boxes

    def shuffle_data(self):
        np.random.shuffle(self.batch_idx)

    def get_batch(self, split, start_idx, batch_img=None, seq_per_img=None):
        batch_img = batch_img or self.batch_img
        seq_per_img = seq_per_img or self.seq_per_img

        feature_batch = []
        label_batch = np.zeros([batch_img * seq_per_img, self.seq_length], dtype='int')

        infos = []
        gts = []

        for i in range(batch_img):
            if split == 'train':
                idx = self.split_idx['train'][self.batch_idx[start_idx*self.batch_img+i]]
            else:
                idx = self.split_idx[split][i]

            # load feature
            file_name = str(self.info['images'][idx]['id']) + '.npz'
            feature = self._load_feature(file_name)
            feature_batch += [feature] * seq_per_img

            # load captions
            idx1 = self.label_start_idx[idx] - 1  # label_start_ix starts from 1
            idx2 = self.label_end_idx[idx] - 1
            ncap = idx2 - idx1 + 1  # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
                for q in range(seq_per_img):
                    idxl = random.randint(idx1, idx2)
                    seq[q, :] = self.h5_label_file['labels'][idxl, :self.seq_length]
            else:
                idxl = random.randint(idx1, idx2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][idxl: idxl + seq_per_img, :self.seq_length]

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, :self.seq_length] = seq

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_idx[idx] - 1: self.label_end_idx[idx]])

            # record associated info as well
            info_dict = {}
            info_dict['idx'] = idx
            info_dict['id'] = self.info['images'][idx]['id']
            info_dict['file_path'] = self.info['images'][idx]['file_path']
            infos.append(info_dict)

        data = {}
        data['features'] = np.stack(feature_batch)
        data['labels'] = label_batch
        data['gts'] = gts
        data['infos'] = infos

        return data

    def get_test_sample(self, sample_idx=None):
        gts = []

        if sample_idx is None:
            sample_idx = np.random.randint(self.num_test_images)
        elif sample_idx >= self.num_test_images:
            print("Out of range in the test dataset!")
            print("Random sampling.")
            sample_idx = np.random.randint(self.num_test_images)
        elif sample_idx < 0:
            sample_idx = np.random.randint(self.num_test_images)

        idx = self.split_idx['test'][sample_idx]

        # load feature and boxes
        file_name = str(self.info['images'][idx]['id']) + '.npz'
        feature = self._load_feature(file_name)
        boxes = self._load_boxes(file_name)

        # load ground-truth captions
        gts.append(self.h5_label_file['labels'][self.label_start_idx[idx] - 1: self.label_end_idx[idx]])

        data = {}
        data['idx'] = idx
        data['id'] = self.info['images'][idx]['id']
        data['file_path'] = self.info['images'][idx]['file_path']
        data['features'] = [feature]
        data['boxes'] = boxes
        data['gts'] = gts

        return data
