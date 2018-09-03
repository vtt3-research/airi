from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import json
import numpy as np

import torch


# For Dense Video Captioning (DVC) Model
class ActivityNet(object):

    def __init__(self, root, data, ids, feat, caps=None, mode='trainval'):
        """
        :param root: data folder path
        :param data: data file contained video dataset (output file from prepro_caps.py)
        :param ids: video name indexes file (output file from prepro_caps.py)
        :param feat: c3d feature file (download)
        :param caps: pre-processed caption to idx vector file (output file from prepro_caps.py)
        :param mode: if mode is 'trainval' then train this class used to train and valid,
                    else this class used to evaluate
        """
        self.mode = mode

        data_file = os.path.join(root, data)
        ids_file = os.path.join(root, ids)
        feat_file = os.path.join(root, feat)
        self.fobj = h5py.File(feat_file, 'r')

        if self.mode == 'trainval':
            caps_file = os.path.join(root, caps)
            self.caps_to_idxvec = h5py.File(caps_file, 'r')

        with open(ids_file) as f:
            self.data_ids = json.load(f)

        with open(data_file) as f:
            data = json.load(f)
            self.dataset = data['video']
            # Options
            if 'word_to_idx' in data:
                self.word_to_idx = data['word_to_idx']
                self.vocab_size = len(self.word_to_idx)
                self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
            else:
                self.word_to_idx = None
                self.vocab_size = 0
                self.idx_to_word = None
            if 'cap_length' in data:
                self.cap_length = data['cap_length']
            else:
                self.cap_length = 0

        self.num_data = len(self.data_ids)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        video_name = self.data_ids[idx]
        feats = self.fobj[video_name]['c3d_features'].value
        # In order to run DVC model, need to features at least 5
        # Padding
        while feats.shape[0] < 5:
            feats = np.concatenate(([feats[0]], feats), axis=0)
            feats = np.concatenate((feats, [feats[-1]]), axis=0)
        duration = self.dataset[video_name]['duration']
        sentences = self.dataset[video_name]['sentences']
        time_stamps = self.dataset[video_name]['timestamps']

        feats = torch.from_numpy(feats).float().permute(1, 0)
        time_boxes = []
        for i in range(len(sentences)):
            start = time_stamps[i][0]/duration
            end = time_stamps[i][1]/duration
            center = (start+end)/2
            width = end-start
            time_boxes.append([center, width, start, end])
        time_boxes = np.asarray(time_boxes)
        time_boxes = torch.from_numpy(time_boxes).float()

        if self.mode == 'trainval':
            cap_to_idxvec = self.caps_to_idxvec[video_name]['vectors'].value
            cap_to_idxvec = torch.from_numpy(cap_to_idxvec)

            # Return video features, ground-truth boxes
            # and ground-truth preprocessed id vectors (captions)
            return feats, time_boxes, cap_to_idxvec

        else:
            # Return video features, ground-truth boxes, duration, time stamps.
            return feats, time_boxes, duration, video_name, \
                   torch.from_numpy(np.asarray(time_stamps)).float()


# For Sentence Generation (SG) Model
class ActivityNetSG(object):

    def __init__(self, root, data, gt_props=None, word_to_idx=None, mode='trainval'):
        """
        :param root: data folder path
        :param data: data file contained averaged features and sentences
                    (output file from prepro_caps.py)
        :param word_to_idx: word to idx file (output file from prepro_caps.py)
        :param gt_props: ground-truth video name and time stamp for each feature/sentence
        :param mode: if mode is 'trainval' then train this class used to train and valid,
                    else this class used to evaluate
        """
        self.mode = mode

        data_file = os.path.join(root, data)

        sg_data = h5py.File(data_file, 'r')
        self.feats = sg_data['features'].value
        self.sents = sg_data['sentences'].value
        self.cap_length = sg_data['cap_length'].value

        # Options
        if word_to_idx is not None:
            word_to_idx_file = os.path.join(root, word_to_idx)
            with open(word_to_idx_file) as f:
                self.word_to_idx = json.load(f)
                self.vocab_size = len(self.word_to_idx)
                self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        else:
            self.word_to_idx = None
            self.vocab_size = 0
            self.idx_to_word = None

        # Ground-truth video name and time stamp for each feature/sentence
        if gt_props is not None:
            gt_proposal_file = os.path.join(root, gt_props)
            with open(gt_proposal_file) as f:
                self.gt_proposals = json.load(f)
        else:
            self.gt_proposals = None

        self.num_data = self.feats.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        feat = self.feats[idx]
        sent = self.sents[idx]

        if self.mode == 'trainval':
            return torch.from_numpy(feat).float(), torch.from_numpy(sent).long()
        else:
            video_name = self.gt_proposals['video_name'][idx]
            timestamp = np.asarray(self.gt_proposals['timestamps'][idx])

            # Return GT proposal features, sentences, video names and time stamps
            return torch.from_numpy(feat).float(), torch.from_numpy(sent).long(), \
                   video_name, torch.from_numpy(timestamp).float()


# For Attribute Detector
class ActivityNetAtts(object):

    def __init__(self, root, data):
        """
        :param root: data folder path
        :param data: data file contained averaged features and classes
                    (output file from prepro_atts.py)
        """
        data_file = os.path.join(root, data)

        att_data = h5py.File(data_file, 'r')
        self.feats = att_data['features'].value
        self.labels = att_data['labels'].value
        self.num_data = self.feats.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        feat = self.feats[idx]
        label = self.labels[idx]
        return torch.from_numpy(feat).float(), label
