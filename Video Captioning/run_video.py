from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import json
import h5py
import numpy as np
import cv2

import torch
from torch.autograd import Variable

from c3d_model import C3D
from model import build_models
from utils import idx_to_sent, box_form


parser = argparse.ArgumentParser(description='Dense Video Captioning')

# Data input settings
parser.add_argument('--root', type=str, default='data/actnet')
parser.add_argument('--train-data', type=str, default='caps_train.json')
parser.add_argument('--video-root', type=str, default=None)
parser.add_argument('--pca-file', type=str, default='data/PCA_activitynet_v1-3.hdf5')

# Model settings
parser.add_argument('--resume-c3d', type=str, default='data/c3d.pickle')
parser.add_argument('--resume-att', type=str, default=None)
parser.add_argument('--resume-dvc', type=str, default=None)
parser.add_argument('-j', '--workers', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=1)

# Parameters
parser.add_argument('--feature-dim', type=int, default=500)
parser.add_argument('--num-class', type=int, default=200)
parser.add_argument('--embedding-dim', type=int, default=1024)
parser.add_argument('--hidden-dim', type=int, default=1024)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--alpha1', type=float, default=0.1)
parser.add_argument('--alpha2', type=float, default=0.1)
parser.add_argument('--lambda0', type=float, default=0.2)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-gpu', action='store_false', help='use gpu (default: True)')
parser.add_argument('--gpu-devices', type=str, default='0')

args = parser.parse_args()


class SizeError(Exception):
    def __str__(self):
        return "Batch size must be set as 1."


class PCA(object):
    def __init__(self, file_name):
        pca = h5py.File(file_name, 'r')
        self.U = pca['data']['U'].value
        self.S = pca['data']['S'].value
        self.x_mean = pca['data']['x_mean'].value

    def transform(self, X, dim):
        if X.shape[1] != self.x_mean.shape[1]:
            raise Exception("Not equal dimension")
        X = X - self.x_mean
        Um = self.U[:dim, :]
        out = np.matmul(X, Um.transpose(1, 0))
        return out


def main():
    global args

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    if not torch.cuda.is_available():
        args.use_gpu = False
    print("USE GPU: {}".format(args.use_gpu))

    # Data load
    data_file = os.path.join(args.root, args.train_data)
    with open(data_file) as f:
        data = json.load(f)
    word_to_idx = data['word_to_idx']
    vocab_size = len(word_to_idx)
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    cap_length = data['cap_length']
    pca = PCA(args.pca_file)

    # Build model
    model_c3d = C3D()
    model_att, model_tep, model_sg, args.scale_ratios = build_models(
        in_c=args.feature_dim, num_class=args.num_class,
        voca_size=vocab_size, caps_length=cap_length,
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, use_gpu=args.use_gpu)
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
                  "for attribute detector module '{}'".format(args.resume_att))
            checkpoint = torch.load(args.resume_att)
            model_att.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_att))
    if args.resume_dvc:
        if os.path.isfile(args.resume_dvc):
            print("=> loading checkpoint "
                  "for DVC with Cross Entropy module '{}'".format(args.resume_dvc))
            checkpoint = torch.load(args.resume_dvc)
            model_tep.load_state_dict(checkpoint['tep_state_dict'])
            model_sg.load_state_dict(checkpoint['sg_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_dvc))

    # Run eval
    run_video(model_c3d, model_att, model_tep, model_sg, pca, idx_to_word)


def run_video(model_c3d, model_att, model_tep, model_sg, pca, idx_to_word):
    # Evaluate mode
    model_c3d.eval()
    model_att.eval()
    model_tep.eval()
    model_sg.eval()

    while True:
        video_file = raw_input("Enter the video path and name:")
        video_file = os.path.join(args.video_root, video_file)
        if video_file == 'Exit':
            break
        if not os.path.exists(video_file):
            continue
        cap = cv2.VideoCapture(video_file)
        num_frame = 0
        frame_list = list()
        frame_list.append(list())
        frame_list.append(list())
        features = None

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                num_frame += 1
                resize_frame = cv2.resize(frame, (112, 112))
                if num_frame <= 8:
                    frame_list[0].append(resize_frame)
                else:
                    frame_list[0].append(resize_frame)
                    frame_list[1].append(resize_frame)
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
                        feature, _ = model_c3d(X)
                        feature = feature.data.cpu().numpy()
                        if features is None:
                            features = feature
                        else:
                            features = np.concatenate((features, feature), axis=0)
            else:
                duration = cap.get(cv2.CAP_PROP_POS_MSEC)
                duration = duration / 1000.0
                cap.release()
                feats = pca.transform(features, 500)
                feats = feats.transpose(1, 0)
                feats = np.expand_dims(feats, axis=0)

        with torch.no_grad():
            data = Variable(torch.from_numpy(feats))
            if args.use_gpu:
                data = data.cuda()

            # Predict proposals
            proposals = model_tep(data)

            # Obtain proposal features with ground-truth proposal
            # using weighted attention(descriptiveness) score
            pos_feats, pos_times = sample_proposal(data, proposals,
                                                   scale_ratios=args.scale_ratios,
                                                   threshold=args.threshold,
                                                   use_gpu=args.use_gpu)

            if args.use_gpu:
                pos_feats = pos_feats.cuda()
            pos_feats = Variable(pos_feats)
            att = model_att(pos_feats)
            if args.use_gpu:
                att = att.cuda()
            att = Variable(att)

            # Generate sentences
            gen_result, _ = model_sg.sample(pos_feats, att, greedy=True)
            gen_sents = idx_to_sent(gen_result, idx_to_word)

            start_times = pos_times[:, 0] * duration
            end_times = pos_times[:, 1] * duration

            for i in range(len(gen_sents)):
                print("[{:.2f}, {:.2f}] {}".format(float(start_times[i]), float(end_times[i]), gen_sents[i][0]))


def sample_proposal(feats, predictions,
                   scale_ratios=np.asarray([1, 1.25, 1.5], dtype=np.float32),
                   threshold=0.7, use_gpu=True):
    loc, des, cls, temporal_dim = predictions

    # Get temporal box
    t_box = box_form(temporal_dim, scale_ratios)

    # Calc clip-level descriptiveness score
    num_feats = feats.shape[2]
    num_anchors = t_box.shape[0]
    feats_center = np.linspace(0, num_feats - 1, num_feats, dtype=np.float32)
    feats_center = (2 * feats_center + 1) / (2.0 * num_feats)

    # for Torch
    x = des[:, :, 0].expand(num_feats, -1)
    y = torch.zeros((num_feats, num_anchors))
    if use_gpu:
        y = y.cuda()
    t_box_start = np.repeat(np.expand_dims(t_box[:, 2], 0), num_feats, axis=0)
    t_box_end = np.repeat(np.expand_dims(t_box[:, 3], 0), num_feats, axis=0)
    feats_center_tile = np.tile(np.expand_dims(feats_center, 1), num_anchors)
    idx = np.logical_and(t_box_start <= feats_center_tile, feats_center_tile <= t_box_end)
    idx = torch.from_numpy(idx.astype(int))
    if use_gpu:
        idx = idx.cuda()
    y[idx == 1] = x[idx == 1]
    clip_des_s = torch.sum(y, 1)
    clip_des_s = clip_des_s/torch.sum(idx, 1).float()

    # Get positive samples
    cand_score = cls.view(-1, 2)[:, 1] + args.lambda0*des.view(-1)
    best_score = torch.max(cand_score).item()
    if best_score < threshold:
        threshold = best_score
    pos_idx = cand_score >= threshold
    predictions_center = t_box[:, 0] + args.alpha1 * t_box[:, 1] * loc[0].data[:, 0]
    predictions_width = t_box[:, 1] * torch.exp(args.alpha2 * loc[0].data[:, 1])
    pos_center = predictions_center[pos_idx]
    pos_width = predictions_width[pos_idx]
    pos_center = pos_center.data.cpu().numpy()
    pos_width = pos_width.data.cpu().numpy()

    # for Torch
    dim_feats = feats.shape[1]
    num_events = pos_center.shape[0]
    pos_feats = torch.zeros(num_events, dim_feats)
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
        x = clip_des_s[start_idx:end_idx]/torch.sum(clip_des_s[start_idx:end_idx])
        y = feats[0, :, start_idx:end_idx]
        pos_feats[i] = torch.sum(x*y, 1)
        time_stamp[i, 0] = start_time
        time_stamp[i, 1] = end_time

    return pos_feats, time_stamp


if __name__ == '__main__':
    main()