from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import json

import torch


# For Sentence Generation (SG) Model
class MSRVTTSG(object):

    def __init__(self, root, data, word_to_idx=None):
        """
        :param root: data folder path
        :param data: data file contained averaged features and sentences
                    (output file from prepro_caps.py)
        :param word_to_idx: word to idx file (output file from prepro_caps.py)
        """
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

        self.num_data = self.feats.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        feat = self.feats[idx]
        sent = self.sents[idx]
        return torch.from_numpy(feat).float(), torch.from_numpy(sent).long()


# For Sentence Generation (SG) Model
class MSRVTTSGRL(object):

    def __init__(self, root, data, video_name, word_to_idx=None):
        """
        :param root: data folder path
        :param data: data file contained averaged features and sentences
                    (output file from prepro_caps.py)
        :param video_name: video name file
        :param word_to_idx: word to idx file (output file from prepro_caps.py)
        """
        data_file = os.path.join(root, data)

        sg_data = h5py.File(data_file, 'r')
        self.feats = sg_data['features'].value
        self.cap_length = sg_data['cap_length'].value
        video_name_file = os.path.join(root, video_name)
        with open(video_name_file) as f:
            self.video_name = json.load(f)

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

        self.num_data = self.feats.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        feat = self.feats[idx]
        video_name = self.video_name[idx]
        return torch.from_numpy(feat).float(), video_name


# For Evaluation of Sentence Generation (SG) Model
class MSRVTTSGEval(object):

    def __init__(self, root, data, gt_props=None, word_to_idx=None):
        data_file = os.path.join(root, data)

        sg_data = h5py.File(data_file, 'r')
        self.feats = sg_data['features'].value
        self.sents = sg_data['sentences'].value
        self.cap_length = sg_data['cap_length'].value

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
        video_name = self.gt_proposals[idx]

        # Return GT proposal features, sentences, video names and time stamps
        return torch.from_numpy(feat).float(), torch.from_numpy(sent).long(), video_name
