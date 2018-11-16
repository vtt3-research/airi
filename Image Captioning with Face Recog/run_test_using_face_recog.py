# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import cv2
import tensorflow as tf
import skimage
import skimage.io
import json
import h5py

from core.model import CaptionGenerator
from core.utils import *

from vtt_face_utils import *
from vtt_face_api import vtt_face_recognize

import caffe


def main(params):

    sys.path.insert(0, os.path.join(params.bottomup_path, 'lib'))
    from fast_rcnn.config import cfg, cfg_from_file
    from fast_rcnn.test import im_detect, _get_blobs
    from fast_rcnn.nms_wrapper import nms

    ###########################
    # CNN : Faster-RCNN setting
    data_path = os.path.join(params.bottomup_path, 'data/genome/1600-400-20')

    # Load classes
    classes = ['__background__']
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    # Load attributes
    attributes = ['__no_attribute__']
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for att in f.readlines():
            attributes.append(att.split(',')[0].lower().strip())

    GPU_ID = params.gpu_id  # if we have multiple GPUs, pick one
    caffe.init_log()
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    net = None
    cfg_from_file(os.path.join(params.bottomup_path, 'experiments/cfgs/faster_rcnn_end2end_resnet.yml'))

    weights = os.path.join(params.bottomup_path, 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel')
    prototxt = os.path.join(params.bottomup_path, 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt')

    net = caffe.Net(prototxt, caffe.TEST, weights=weights)

    conf_thresh = 0.4
    min_boxes = params.num_objects
    max_boxes = params.num_objects
    ###########################

    ###########################
    # RNN : Caption generation setting
    # load json file
    label_info = json.load(open(params.input_labels))
    word_to_idx = label_info['word_to_idx']

    # load h5 file
    caps_info = h5py.File(params.input_caps, 'r', driver='core')
    seq_length = caps_info['labels'].shape[1]

    # GPU options
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # build a graph to sample captions
    graph_gen_cap = tf.Graph()
    sess_gen_cap = tf.Session(graph=graph_gen_cap, config=config)
    with graph_gen_cap.as_default():
        model = CaptionGenerator(word_to_idx, num_features=params.num_objects, dim_feature=params.dim_features,
                                 dim_embed=params.dim_word_emb, dim_hidden=params.rnn_hid_size,
                                 dim_attention=params.att_hid_size, n_time_step=seq_length - 1)
        alphas, sampled_captions = model.build_sampler(max_len=params.max_len)
        saver1 = tf.train.Saver()
        saver1.restore(sess_gen_cap, params.test_model)
    tf.reset_default_graph()
    ############################

    ###########################
    # Face : Replacer
    name_replacer = NameReplacer(model.idx_to_word, params.score_thr)
    ############################

    ###########################
    # Run Image Captioning with face detection

    while True:
        full_fname = raw_input("Enter the image path and name:")
        if full_fname == 'Exit':
            break
        if not os.path.exists(full_fname):
            print("Not Exist File : {}".format(full_fname))
            continue

        ###########################
        # Object Detection
        im = cv2.imread(full_fname)
        scores, boxes, attr_scores, rel_scores = im_detect(net, im)

        # Keep the original boxes, don't worry about the regression bbox outputs
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        blobs, im_scales = _get_blobs(im, None)

        cls_boxes = rois[:, 1:5] / im_scales[0]
        cls_prob = net.blobs['cls_prob'].data
        attr_prob = net.blobs['attr_prob'].data
        pool5 = net.blobs['pool5_flat'].data

        # Keep only the best detections
        max_conf = np.zeros((rois.shape[0]))
        for cls_ind in range(1, cls_prob.shape[1]):
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = np.array(nms(dets, cfg.TEST.NMS))
            max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

        keep_boxes = np.where(max_conf >= conf_thresh)[0]
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf)[::-1][:max_boxes]

        feats = pool5[keep_boxes]
        ############################

        ###########################
        # Caption generation using CNN features
        feed_dict = {model.features: [feats]}
        alps, sam_cap = sess_gen_cap.run([alphas, sampled_captions], feed_dict)
        decoded = decode_captions(sam_cap, model.idx_to_word)
        ############################

        ###########################
        # Name replacer
        name_list, conf_list, roi_list = vtt_face_recognize(full_fname, params.url, params.post_data)
        replace_decoded, words = name_replacer.name_replace_caps(sam_cap, alps, cls_boxes,
                                                                 name_list, conf_list, roi_list)
        print("Original caption : %s" % decoded[0])
        print("Replaced caption : %s" % replace_decoded[0])
        ############################

        ###########################
        # Showing
        img = skimage.io.imread(full_fname)
        img = skimage.img_as_float(img)
        boxes = cls_boxes[keep_boxes]
        boxes = boxes.astype(int)

        # draw attention map
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(3, 6, 1)
        ax.imshow(img)
        plt.axis('off')

        # Plot images with attention weights
        words = words[0]
        for t in range(len(words)):
            if t > 16:
                break
            if words[t] == '<BLANK>':
                continue
            alphamap = np.zeros((img.shape[0], img.shape[1]))
            for b in range(boxes.shape[0]):
                alphamap[boxes[b, 1]:boxes[b, 3], boxes[b, 0]:boxes[b, 2]] += alps[0, t, b]
            max_idx = np.argmax(alps[0, t, :])
            att_img = np.dstack((img, alphamap))
            ax = fig.add_subplot(3, 6, t + 2)
            plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
            ax.imshow(att_img)
            ax.add_patch(patches.Rectangle((boxes[max_idx, 0], boxes[max_idx, 1]),
                                           boxes[max_idx, 2] - boxes[max_idx, 0],
                                           boxes[max_idx, 3] - boxes[max_idx, 1],
                                           linewidth=1, edgecolor='r', facecolor='none'))
            plt.axis('off')

        fig.tight_layout()
        plt.show()
        ############################


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Data inputs
    parser.add_argument('--input_labels', default='data/vtt/vtt_labels.json',
                        help='Input labels data file (generate from prepro_captions.py)')
    parser.add_argument('--input_caps', default='data/vtt/vtt_captions.h5',
                        help='Input captions data file (generate from prepro_captions.py)')
    parser.add_argument('--bottomup_path', default='bottom-up-attention',
                        help='Installed bottom-up object detection path')
    parser.add_argument('--test_model', default='model/model_rl-20')

    # Informations for data
    parser.add_argument('--num_objects', default=36, type=int, help='number of objects(features) in each image for CNN')
    parser.add_argument('--dim_features', default=2048, type=int, help='dimension of the image feature')
    parser.add_argument('--max_len', default=16, type=int, help='maximum caption length')

    # Parameters for network
    parser.add_argument('--dim_word_emb', default=1000, type=int,
                        help='dimension for the word embedding in the caption model')
    parser.add_argument('--att_hid_size', default=512, type=int, help='size of the hidden unit in the attention layer')
    parser.add_argument('--rnn_hid_size', default=1000, type=int, help='size of the hidden unit in the LSTM')

    parser.add_argument('--score_thr', default=0.5, type=float, help='Name replacer threshold')
    parser.add_argument('--gpu_id', default=0, type=int)

    # For face recognition
    parser.add_argument('--url', default=None)
    parser.add_argument('--post_data', default=None)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
