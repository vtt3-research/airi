# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import cv2
import numpy as np

import torch
from torch.autograd import Variable

from new_model import build_model
from new_utils import sample_proposal, idx_to_sent

import threading
from flask import Flask, Response, request, render_template


parser = argparse.ArgumentParser()

# For client-server
parser.add_argument('--host', type=str, default='127.0.0.1', help='host name')
parser.add_argument('--port', type=int, default=3334, help='port number')

# Data input settings
parser.add_argument('--root', type=str, default='data/actnet_msrvtt')
parser.add_argument('--word_to_idx', type=str, default='word_to_idx.json')

# Model settings
parser.add_argument('--resume-c3d', type=str, default='models/c3d.pickle')
parser.add_argument('--resume-att', type=str, default='models/att_model.pth.tar')
parser.add_argument('--resume-tep', type=str, default='models/tep_model.pth.tar')
parser.add_argument('--resume-sg', type=str, default='models/sg_model.pth.tar')
parser.add_argument('-j', '--workers', type=int, default=1)

# Parameters
parser.add_argument('--cap-length', type=int, default=18)
parser.add_argument('--num-class', type=int, default=201)
parser.add_argument('--num-feats', type=int, default=36)
parser.add_argument('--embedding-dim', type=int, default=1024)
parser.add_argument('--hidden-dim', type=int, default=1024)
parser.add_argument('--alpha1', type=float, default=0.1)
parser.add_argument('--alpha2', type=float, default=0.1)
parser.add_argument('--pos_thr', type=float, default=0.4)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-gpu', action='store_false', help='use gpu (default: True)')
parser.add_argument('--gpu-devices', type=str, default='0')

args = parser.parse_args()

sem = threading.Semaphore(1)


def main():
    global args

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    if not torch.cuda.is_available():
        args.use_gpu = False
    print("USE GPU: {}".format(args.use_gpu))

    # Data load
    word_to_idx_file = os.path.join(args.root, args.word_to_idx)
    with open(word_to_idx_file) as f:
        word_to_idx = json.load(f)
    vocab_size = len(word_to_idx)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    cap_length = args.cap_length

    # Build model
    model_c3d, model_att, model_tep, model_sg, args.scale_ratios = build_model(
        num_class=args.num_class, voca_size=vocab_size,
        caps_length=cap_length, embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim, use_gpu=args.use_gpu)
    if args.use_gpu:
        model_c3d = model_c3d.cuda()
        model_att = model_att.cuda()
        model_tep = model_tep.cuda()
        model_sg = model_sg.cuda()

    # Load resume from a checkpoint
    if args.resume_c3d:
        if os.path.isfile(args.resume_c3d):
            print("=> loading checkpoint "
                  "for C3D module '{}'".format(args.resume_c3d))
            checkpoint = torch.load(args.resume_c3d)
            model_c3d.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_c3d))
    if args.resume_att:
        if os.path.isfile(args.resume_att):
            print("=> loading checkpoint "
                  "for Attribute Detector module '{}'".format(args.resume_att))
            checkpoint = torch.load(args.resume_att)
            model_att.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_att))
    if args.resume_tep:
        if os.path.isfile(args.resume_tep):
            print("=> loading checkpoint "
                  "for temporal event proposals module '{}'".format(args.resume_tep))
            checkpoint = torch.load(args.resume_tep)
            model_tep.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_tep))
    if args.resume_sg:
        if os.path.isfile(args.resume_sg):
            print("=> loading checkpoint "
                  "for sentence generation module '{}'".format(args.resume_sg))
            checkpoint = torch.load(args.resume_sg)
            model_sg.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_sg))

    # Evaluate mode
    model_c3d.eval()
    model_att.eval()
    model_tep.eval()
    model_sg.eval()

    while True:
        video_file = raw_input("Enter the video path and name:")

        if video_file == 'Exit' or video_file == 'exit':
            break
        if not os.path.exists(video_file):
            continue

        cap = cv2.VideoCapture(video_file)
        num_frame = 0
        frame_list = list()
        frame_list.append(list())
        frame_list.append(list())
        features = None

        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    num_frame += 1
                    h, w, c = frame.shape

                    # crop to center region
                    if h > w:
                        pad = int((h - w) / 2.0)
                        center_img = frame[pad:w + pad, :, :]
                    elif w > h:
                        pad = int((w - h) / 2.0)
                        center_img = frame[:, pad:h + pad, :]
                    else:
                        center_img = frame

                    # resize for network input
                    resize_img = cv2.resize(center_img, (112, 112))

                    # extract feature
                    if num_frame <= 8:
                        frame_list[0].append(resize_img)
                    else:
                        frame_list[0].append(resize_img)
                        frame_list[1].append(resize_img)
                        if (num_frame % 8) == 0:
                            if (num_frame % 16) == 0:
                                clip = np.asarray(frame_list[0], dtype=np.float32)
                                frame_list[0] = list()
                            else:
                                clip = np.asarray(frame_list[1], dtype=np.float32)
                                frame_list[1] = list()
                            clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
                            clip = np.expand_dims(clip, axis=0)  # batch axis
                            X = Variable(torch.from_numpy(clip))
                            if args.use_gpu:
                                X = X.cuda()
                            feature = model_c3d(X)
                            feature = feature.data.cpu().numpy()
                            if features is None:
                                features = feature
                            else:
                                features = np.concatenate((features, feature), axis=0)
                else:
                    duration = cap.get(cv2.CAP_PROP_POS_MSEC)
                    duration = duration / 1000.0
                    cap.release()

            if features is None:
                continue

            features = Variable(torch.from_numpy(features))
            if args.use_gpu:
                features = features.cuda()

            while features.shape[0] < 5:
                torch.cat((features[0].unsqueeze(0), features), dim=0)
                torch.cat((features, features[-1].unsqueeze(0)), dim=0)
            features = features.transpose(1, 0).unsqueeze(0)

            # Predict proposals
            proposals = model_tep(features)

            pos_feats, pos_times = sample_proposal(features, proposals, args.alpha1,
                                                   args.alpha2, args.num_feats,
                                                   scale_ratios=args.scale_ratios,
                                                   threshold=args.pos_thr,
                                                   duration=duration)

            if args.use_gpu:
                pos_feats = pos_feats.cuda()

            # Attribute detection
            pos_att = model_att(torch.mean(pos_feats, 2))

            # Generate sentences
            gen_result, _ = model_sg.sample(pos_feats, pos_att, greedy=True)
            gen_sents = idx_to_sent(gen_result, idx_to_word)

            start_times = pos_times[:, 0] * duration
            end_times = pos_times[:, 1] * duration
            sort_idx = np.argsort(start_times)

            print("num_caps :", len(gen_sents))
            for i in range(len(gen_sents)):
                print("[{:.2f}~{:.2f}] {}\n".format(float(start_times[sort_idx[i]]),
                                                    float(end_times[sort_idx[i]]),
                                                    gen_sents[sort_idx[i]][0]))

if __name__ == '__main__':
    main()