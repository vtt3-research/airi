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


def _build_vocab(caps, ids, threshold=1, max_len=15):
    counter = Counter()
    for idx in ids:
        for sent in caps[idx]['sentences']:
            sent = _process_caption(sent)
            words = sent.split(' ')
            sent_len = 0
            for w in words:
                counter[w] += 1
                sent_len += 1
                if sent_len >= max_len:
                    break

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
    for i, id in enumerate(caps):
        cap = caps[id]
        caption_vectors = []
        for sent in cap['sentences']:
            cap_vec = []
            sent = _process_caption(sent)
            words = sent.split(' ')
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
            caption_vectors.append(cap_vec)
        caps[id]['vectors'] = caption_vectors


def _build_features(caps, feats):
    sg_feats = list()
    sg_vecs = list()
    sg_names = list()
    sg_timestamps = list()

    count = 0
    for i, v_name in enumerate(caps):
        cap = caps[v_name]
        v_feats = feats[v_name]['c3d_features'].value
        duration = cap['duration']
        for j in range(len(cap['vectors'])):
            start, end = cap['timestamps'][j]
            start = v_feats.shape[0] * start / duration
            end = v_feats.shape[0] * end / duration
            start = np.round(start).astype(int)
            end = np.round(end).astype(int)
            if start == end:
                if end != v_feats.shape[0]:
                    end = end + 1
                else:
                    start = start - 1
            sg_feat = v_feats[start:end]
            sg_feat = np.sum(sg_feat, axis=0) / float(end - start)
            sg_vec = cap['vectors'][j]
            sg_feats.append(sg_feat)
            sg_vecs.append(sg_vec)
            sg_names.append(v_name)
            sg_timestamps.append(cap['timestamps'][j])

        count += 1
        if count % 1000 == 0:
            print("Processed {} ...".format(count))

    print("Finished")
    sg_feats = np.asarray(sg_feats)
    sg_vecs = np.asarray(sg_vecs)
    return sg_feats, sg_vecs, sg_names, sg_timestamps


def remove_nonexistent_idx(ids, caps):
    ids_fix = []
    for i in range(len(ids)):
        if ids[i] in caps:
            ids_fix.append(ids[i])
    return ids_fix


def main(params):
    feats = h5py.File(params.input_feats, 'r')

    # With train dataset
    train_ids = json.load(open(os.path.join(params.input_root, params.input_train_ids), 'r'))
    train_caps = json.load(open(os.path.join(params.input_root, params.input_train_caps), 'r'))

    train_ids_fix = remove_nonexistent_idx(train_ids, train_caps)
    word_to_idx = _build_vocab(train_caps, train_ids_fix, params.word_count_threshold)
    _build_onehot_caption_vector(train_caps, word_to_idx, params.max_len)
    train_feats, train_vecs, _, _ = _build_features(train_caps, feats)

    # With validation dataset
    val_ids = json.load(open(os.path.join(params.input_root, params.input_val_ids), 'r'))
    val_caps1 = json.load(open(os.path.join(params.input_root, params.input_val_caps1), 'r'))
    val_caps2 = json.load(open(os.path.join(params.input_root, params.input_val_caps2), 'r'))

    val_ids_fix1 = remove_nonexistent_idx(val_ids, val_caps1)
    val_ids_fix2 = remove_nonexistent_idx(val_ids, val_caps2)
    _build_onehot_caption_vector(val_caps1, word_to_idx, params.max_len)
    _build_onehot_caption_vector(val_caps2, word_to_idx, params.max_len)
    val_feats1, val_vecs1, val_vnames1, val_timestamps1 = _build_features(val_caps1, feats)
    val_feats2, val_vecs2, val_vnames2, val_timestamps2 = _build_features(val_caps2, feats)

    # Save data for Sentence Generator training
    print("Save file for Sentence Generator...")
    json.dump(word_to_idx, open(os.path.join(params.output_root, params.output_wordtoidx), 'w'))
    f_lb = h5py.File(os.path.join(params.output_root, params.output_train_sg), "w")
    f_lb.create_dataset('cap_length', data=(params.max_len+2))
    f_lb.create_dataset('features', data=train_feats)
    f_lb.create_dataset('sentences', data=train_vecs)
    f_lb.close()

    # Save data for Sentence Generator validation
    f_lb = h5py.File(os.path.join(params.output_root, params.output_val_sg), "w")
    f_lb.create_dataset('cap_length', data=(params.max_len+2))
    f_lb.create_dataset('features', data=val_feats1)
    f_lb.create_dataset('sentences', data=val_vecs1)
    f_lb.close()
    out = {}
    out['video_name'] = val_vnames1
    out['timestamps'] = val_timestamps1
    json.dump(out, open(os.path.join(params.output_root, params.output_val_sg_eval), 'w'))

    f_lb = h5py.File(os.path.join(params.output_root, params.output_val_sg2), "w")
    f_lb.create_dataset('cap_length', data=(params.max_len+2))
    f_lb.create_dataset('features', data=val_feats2)
    f_lb.create_dataset('sentences', data=val_vecs2)
    f_lb.close()
    f_lb.close()
    out = {}
    out['video_name'] = val_vnames2
    out['timestamps'] = val_timestamps2
    json.dump(out, open(os.path.join(params.output_root, params.output_val_sg_eval2), 'w'))

    # Save data for Dense Video Captioning training
    print("Save file for Dense Video Captioning...")
    f_lb = h5py.File(os.path.join(params.output_root, params.output_train_vecs), "w")
    for i, id in enumerate(train_caps):
        grp = f_lb.create_group(id)
        grp.create_dataset("vectors", data=train_caps[id]['vectors'])
    f_lb.close()
    json.dump(train_ids_fix, open(os.path.join(params.output_root, params.output_train_ids), 'w'))
    out = {}
    out['word_to_idx'] = word_to_idx
    out['cap_length'] = params.max_len + 2
    out['video'] = {}
    for i, video_name in enumerate(train_ids_fix):
        out['video'][video_name] = {}
        out['video'][video_name]['duration'] = train_caps[video_name]['duration']
        out['video'][video_name]['sentences'] = train_caps[video_name]['sentences']
        out['video'][video_name]['timestamps'] = train_caps[video_name]['timestamps']
    json.dump(out, open(os.path.join(params.output_root, params.output_train_caps), 'w'))

    # Save data for Dense Video Captioning validation
    f_lb = h5py.File(os.path.join(params.output_root, params.output_val_vecs1), "w")
    for i, id in enumerate(val_caps1):
        grp = f_lb.create_group(id)
        grp.create_dataset("vectors", data=val_caps1[id]['vectors'])
    f_lb.close()
    json.dump(val_ids_fix1, open(os.path.join(params.output_root, params.output_val_ids1), 'w'))
    out = {}
    out['video'] = {}
    for i, video_name in enumerate(val_ids_fix1):
        out['video'][video_name] = {}
        out['video'][video_name]['duration'] = val_caps1[video_name]['duration']
        out['video'][video_name]['sentences'] = val_caps1[video_name]['sentences']
        out['video'][video_name]['timestamps'] = val_caps1[video_name]['timestamps']
    json.dump(out, open(os.path.join(params.output_root, params.output_val_caps1), 'w'))

    # Save data for Dense Video Captioning validation 2
    f_lb = h5py.File(os.path.join(params.output_root, params.output_val_vecs2), "w")
    for i, id in enumerate(val_caps2):
        grp = f_lb.create_group(id)
        grp.create_dataset("vectors", data=val_caps2[id]['vectors'])
    f_lb.close()
    json.dump(val_ids_fix2, open(os.path.join(params.output_root, params.output_val_ids2), 'w'))
    out = {}
    out['video'] = {}
    for i, video_name in enumerate(val_ids_fix2):
        out['video'][video_name] = {}
        out['video'][video_name]['duration'] = val_caps2[video_name]['duration']
        out['video'][video_name]['sentences'] = val_caps2[video_name]['sentences']
        out['video'][video_name]['timestamps'] = val_caps2[video_name]['timestamps']
    json.dump(out, open(os.path.join(params.output_root, params.output_val_caps2), 'w'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Input data files
    parser.add_argument('--input_feats', default='./data/actnet/sub_activitynet_v1-3.c3d.hdf5',
                        help='input ActivityNet200 c3d feature file (hdf5 file)')
    parser.add_argument('--input_root', default='./data/captions')
    parser.add_argument('--input_train_ids', default='train_ids.json',
                        help='input train video name list (json file)')
    parser.add_argument('--input_train_caps', default='train.json',
                        help='input train video caption (json file)')
    parser.add_argument('--input_val_ids', default='val_ids.json',
                        help='input validation video name list (json file)')
    parser.add_argument('--input_val_caps1', default='val_1.json',
                        help='input validation video caption (json file)')
    parser.add_argument('--input_val_caps2', default='val_2.json',
                        help='input validation video caption (json file)')

    # Output data root
    parser.add_argument('--output_root', default='./data/actnet')

    # Output data file for Sentence Generator
    parser.add_argument('--output_wordtoidx', default='sg_word_to_idx.json',
                        help='word-to-idx file (json file)')
    parser.add_argument('--output_train_sg', default='sg_train.hdf5',
                        help='output file with features and sentences for SG pre-training (hdf5 file)')
    parser.add_argument('--output_val_sg', default='sg_val.hdf5',
                        help='output file with features and sentences for SG pre-training (hdf5 file)')
    parser.add_argument('--output_val_sg_eval', default='sg_val_eval.json',
                        help='output file with ground-truth video name and proposals (json file)')
    parser.add_argument('--output_val_sg2', default='sg_val2.hdf5',
                        help='output file with features and sentences for SG pre-training (hdf5 file)')
    parser.add_argument('--output_val_sg_eval2', default='sg_val_eval2.json',
                        help='output file with ground-truth video name and proposals (json file)')

    # Output data file for Dense Video Captioning
    parser.add_argument('--output_train_ids', default='caps_train_ids.json',
                        help='output file with fixed train video name list (json file). '
                             'video name without caption removed list')
    parser.add_argument('--output_train_vecs', default='caps_train_vecs.hdf5',
                        help='output file with index vectors represent as captions for training (hdf5 file)')
    parser.add_argument('--output_train_caps', default='caps_train.json',
                        help='output file with train video data and word to vector (json file)')
    parser.add_argument('--output_val_ids1', default='caps_val_ids.json',
                        help='output file with fixed validation video name list (json file). '
                             'video name without caption removed list')
    parser.add_argument('--output_val_ids2', default='caps_val_ids2.json',
                        help='output file with fixed validation video name list (json file). '
                             'video name without caption removed list')
    parser.add_argument('--output_val_vecs1', default='caps_val_vecs.hdf5',
                        help='output file with index vectors represent as captions for validation (hdf5 file)')
    parser.add_argument('--output_val_vecs2', default='caps_val_vecs2.hdf5',
                        help='output file with index vectors represent as captions for validation (hdf5 file)')
    parser.add_argument('--output_val_caps1', default='caps_val.json',
                        help='output file with validation video data (json file)')
    parser.add_argument('--output_val_caps2', default='caps_val2.json',
                        help='output file with validation video data (json file)')

    # Parameters
    parser.add_argument('--max_len', default=2, type=int, help='max length of a caption')
    parser.add_argument('--word_count_threshold', default=3, type=int,
                        help='words that do not occur at least this number, filtering in vocabulary')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
