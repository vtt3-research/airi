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


def _process_caption(caption):
    caption = caption.lower()
    caption = caption.replace('.', '').replace(
        ',', '').replace("'", "").replace('"', '')
    caption = caption.replace('&', 'and').replace(
        '(', '').replace(")", "").replace('-', ' ')
    caption = " ".join(caption.split())
    return caption


def _build_vocab(caps, threshold=1):
    counter = Counter()
    for i in range(len(caps['sentences'])):
        sent = caps['sentences'][i]['caption']
        sent = _process_caption(sent)
        words = sent.split(' ')
        for w in words:
            counter[w] += 1

    vocab = [word for word in counter if counter[word] >= threshold]
    print ("Filtered {} words to {} words with word count threshold {}.".format(
        len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    return word_to_idx


def _build_onehot_caption_vector(caps, word_to_idx, max_len=15):
    for i in range(len(caps['sentences'])):
        sent = caps['sentences'][i]['caption']
        sent = _process_caption(sent)
        words = sent.split(' ')
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for w in words:
            if w in word_to_idx:
                cap_vec.append(word_to_idx[w])
            else:
                cap_vec.append(word_to_idx['<UNK>'])
            if len(cap_vec) > max_len:
                break
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it
        # fixed-size vector
        if len(cap_vec) < (max_len + 2):
            for j in range(max_len + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])
        caps['sentences'][i]['vector'] = cap_vec


def _build_sg_feats(caps, feats, train_ids, val_ids):
    sg_train_feats = list()
    sg_train_vecs = list()
    sg_train_rl_feats = list()
    sg_train_rl_video_id = list()
    sg_val_feats = list()
    sg_val_vecs = list()
    sg_val_video_id = list()

    count = 0
    for i in range(len(caps['sentences'])):
        video_id = caps['sentences'][i]['video_id']
        if video_id in train_ids:
            sg_train_feats.append(feats[video_id].value)
            sg_train_vecs.append(caps['sentences'][i]['vector'])
            if video_id not in sg_train_rl_video_id:
                sg_train_rl_feats.append(feats[video_id].value)
                sg_train_rl_video_id.append(video_id)
        elif video_id in val_ids:
            if video_id not in sg_val_video_id:
                sg_val_feats.append(feats[video_id].value)
                sg_val_vecs.append(caps['sentences'][i]['vector'])
                sg_val_video_id.append(video_id)

        count += 1
        if count % 1000 == 0:
            print("Processed {} ...".format(count))

    print("Finished : {} trains, {} vals".format(len(sg_train_feats), len(sg_val_feats)))
    sg_train_feats = np.asarray(sg_train_feats)
    sg_train_vecs = np.asarray(sg_train_vecs)
    sg_val_feats = np.asarray(sg_val_feats)
    sg_val_vecs = np.asarray(sg_val_vecs)
    return sg_train_feats, sg_train_vecs, sg_train_rl_feats, sg_train_rl_video_id, \
           sg_val_feats, sg_val_vecs, sg_val_video_id


def _resort_sentence(caps, ids, flag='val'):
    video_sents = {}
    for i in range(len(caps['sentences'])):
        video_id = caps['sentences'][i]['video_id']
        if flag == 'val' and video_id in ids:
            caption = caps['sentences'][i]['caption']

            if not video_id in video_sents:
                video_sents[video_id] = []

            temp = {}
            temp['sentence'] = caption
            video_sents[video_id].append(temp)
        elif flag == 'train' and video_id in ids:
            caption = caps['sentences'][i]['caption']

            if not video_id in video_sents:
                video_sents[video_id] = []

            temp = {}
            temp['sentence'] = caption
            video_sents[video_id].append(temp)

    return video_sents


def _build_att_feats(caps, feats, train_ids, val_ids):
    att_train_feats = list()
    att_train_labels = list()
    att_val_feats = list()
    att_val_labels = list()

    count = 0
    for i in range(len(caps['videos'])):
        video_id = caps['videos'][i]['video_id']
        if video_id in train_ids:
            att_train_feats.append(feats[video_id].value)
            att_train_labels.append(caps['videos'][i]['category'])
        elif video_id in val_ids:
            att_val_feats.append(feats[video_id].value)
            att_val_labels.append(caps['videos'][i]['category'])
        else:
            print('pass :', video_id)

        count += 1
        if count % 1000 == 0:
            print("Processed {} ...".format(count))

    print("Finished")
    att_train_feats = np.asarray(att_train_feats)
    att_train_labels = np.asarray(att_train_labels)
    att_val_feats = np.asarray(att_val_feats)
    att_val_labels = np.asarray(att_val_labels)

    return att_train_feats, att_train_labels, att_val_feats, att_val_labels


def main(params):
    data = json.load(open(os.path.join(params.input_root, params.input_info), 'r'))
    feats = h5py.File(os.path.join(params.input_root, params.input_feats), 'r')
    print("Loaded Data")

    train_ids = list()
    val_ids = list()
    for i in range(len(data['videos'])):
        if data['videos'][i]['split'] == 'train':
            train_ids.append(data['videos'][i]['video_id'])
        elif data['videos'][i]['split'] == 'validate':
            val_ids.append(data['videos'][i]['video_id'])
        else:
            print('pass :', data['videos'][i]['split'])

    word_to_idx = _build_vocab(data, params.word_count_threshold)
    _build_onehot_caption_vector(data, word_to_idx, params.max_len)
    train_att_feats, train_att_labels, val_att_feats, val_att_labels = _build_att_feats(data, feats, train_ids, val_ids)
    train_sg_feats, train_sg_vecs, train_sg_rl_feats, train_sg_rl_video_id, \
    val_sg_feats, val_sg_vects, val_sg_video_id = _build_sg_feats(data, feats, train_ids, val_ids)
    train_sents = _resort_sentence(data, train_ids, 'train')
    val_sents = _resort_sentence(data, val_ids, 'val')

    # Save data for Attribute Detector
    print("Save file for Attribute Detector...")
    f_lb = h5py.File(os.path.join(params.output_root, params.output_train_att), "w")
    f_lb.create_dataset('features', data=train_att_feats)
    f_lb.create_dataset('labels', data=train_att_labels)
    f_lb.close()
    f_lb = h5py.File(os.path.join(params.output_root, params.output_val_att), "w")
    f_lb.create_dataset('features', data=val_att_feats)
    f_lb.create_dataset('labels', data=val_att_labels)
    f_lb.close()

    # Save data for Sentence Generator
    print("Save file for Sentence Generator...")
    json.dump(word_to_idx, open(os.path.join(params.output_root, params.output_wordtoidx), 'w'))
    f_lb = h5py.File(os.path.join(params.output_root, params.output_train_sg), "w")
    f_lb.create_dataset('cap_length', data=(params.max_len + 2))
    f_lb.create_dataset('features', data=train_sg_feats)
    f_lb.create_dataset('sentences', data=train_sg_vecs)
    f_lb.close()

    f_lb = h5py.File(os.path.join(params.output_root, params.output_train_sg_rl), "w")
    f_lb.create_dataset('cap_length', data=(params.max_len + 2))
    f_lb.create_dataset('features', data=train_sg_rl_feats)
    f_lb.close()
    json.dump(train_sg_rl_video_id, open(os.path.join(params.output_root, params.output_train_sg_vid), 'w'))
    json.dump(train_sents, open(os.path.join(params.output_root, params.output_train_sg_eval), 'w'))

    f_lb = h5py.File(os.path.join(params.output_root, params.output_val_sg), "w")
    f_lb.create_dataset('cap_length', data=(params.max_len + 2))
    f_lb.create_dataset('features', data=val_sg_feats)
    f_lb.create_dataset('sentences', data=val_sg_vects)
    f_lb.close()
    json.dump(val_sg_video_id, open(os.path.join(params.output_root, params.output_val_sg_vid), 'w'))
    json.dump(val_sents, open(os.path.join(params.output_root, params.output_val_sg_eval), 'w'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Input data files
    parser.add_argument('--input_root', default='./data/MSRVTT')
    parser.add_argument('--input_info', default='videodatainfo.json',
                        help='input MSR-VTT video information file (json file)')
    parser.add_argument('--input_feats', default='msrvtt_features.hdf5',
                        help='input MSR-VTT video feature file (h5py file)')

    # Output data root
    parser.add_argument('--output_root', default='./data/MSRVTT')

    # Output data file for Attribute detector
    parser.add_argument('--output_train_att', default='att_train.hdf5',
                        help='output file with features and labels for training (hdf5 file)')
    parser.add_argument('--output_val_att', default='att_val.hdf5',
                        help='output file with features and labels for validation (hdf5 file)')

    # Output data file for Sentence Generator
    parser.add_argument('--output_wordtoidx', default='sg_word_to_idx.json',
                        help='word-to-idx file (json file)')
    parser.add_argument('--output_train_sg', default='sg_train.hdf5',
                        help='output file with features and sentences for SG training (hdf5 file)')
    parser.add_argument('--output_train_sg_rl', default='sg_train_rl.hdf5',
                        help='output file with features for SG reinforcement learning (hdf5 file)')
    parser.add_argument('--output_train_sg_vid', default='sg_train_vid.json',
                        help='output file with video id (json file)')
    parser.add_argument('--output_train_sg_eval', default='sg_train_eval.json',
                        help='output file with training sentences (json file)')
    parser.add_argument('--output_val_sg', default='sg_val.hdf5',
                        help='output file with features and sentences for SG training (hdf5 file)')
    parser.add_argument('--output_val_sg_vid', default='sg_val_vid.json',
                        help='output file with video id (json file)')
    parser.add_argument('--output_val_sg_eval', default='sg_val_eval.json',
                        help='output file with validation sentences (json file)')

    # Parameters
    parser.add_argument('--max_len', default=25, type=int, help='max length of a caption')
    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='words that do not occur at least this number, filtering in vocabulary')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
