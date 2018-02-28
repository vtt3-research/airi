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


def _build_vocab(imgs, threshold=1):
    counter = Counter()

    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counter[w] += 1

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' %
           (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2, u'<UNK>': 3}
    idx = 4
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    return word_to_idx


def _build_new_caption_vector(imgs, word_to_idx, max_len=15):
    len_imgs = len(imgs)
    caption_vectors = []
    label_start_idx = np.zeros(len_imgs, dtype='uint32')
    label_end_idx = np.zeros(len_imgs, dtype='uint32')

    count = 1
    for i, img in enumerate(imgs):
        n = len(img['sentences'])
        for sent in img['sentences']:
            cap_vec = []
            cap_vec.append(word_to_idx['<START>'])
            for word in sent['tokens']:
                if word in word_to_idx:
                    cap_vec.append(word_to_idx[word])
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
        label_start_idx[i] = count
        label_end_idx[i] = count + n - 1
        count += n

    return np.array(caption_vectors), label_start_idx, label_end_idx


def main(params):
    imgs = json.load(open(params.input_file, 'r'))
    imgs = imgs['images']

    word_to_idx = _build_vocab(imgs, params.word_count_threshold)
    labels, label_start_idx, label_end_idx = _build_new_caption_vector(imgs, word_to_idx, params.max_len)

    f_lb = h5py.File(params.output_cap, "w")
    f_lb.create_dataset("labels", dtype='uint32', data=labels)
    f_lb.create_dataset("label_start_idx", dtype='uint32', data=label_start_idx)
    f_lb.create_dataset("label_end_idx", dtype='uint32', data=label_end_idx)
    f_lb.close()

    out = {}
    out['word_to_idx'] = word_to_idx
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img:
            jimg['file_path'] = os.path.join(img['filepath'], img['filename'])
        if 'cocoid' in img:
            jimg['id'] = img['cocoid']
        out['images'].append(jimg)

    json.dump(out, open(params.output_label, 'w'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='./data/coco/dataset_coco.json', help='input file')
    parser.add_argument('--output_cap', default='./data/coco/coco_captions.h5', help='output h5 file')
    parser.add_argument('--output_label', default='./data/coco/coco_labels.json', help='output json file')
    parser.add_argument('--max_len', default=15, type=int, help='max length of a caption')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='words that do not occur at least this number, filtering in vocabulary')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
