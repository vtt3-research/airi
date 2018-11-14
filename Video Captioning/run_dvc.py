from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import shutil
import argparse
import json

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_manager import ActivityNet
from model import build_models
from loss import DVCLoss
from rewards import init_eval_metric, get_sc_reward, compute_meteor_score, expand_reward
from utils import clip_des_score, get_gt_proposal, idx_to_sent

sys.path.insert(0, './densevid_eval')
from evaluate import ANETcaptions


parser = argparse.ArgumentParser(description='Dense Video Captioning')

# Data input settings
parser.add_argument('--root', type=str, default='data/actnet')
parser.add_argument('--feature-set', type=str, default='sub_activitynet_v1-3.c3d.hdf5')
parser.add_argument('--train-data', type=str, default='caps_train.json')
parser.add_argument('--train-ids', type=str, default='caps_train_ids.json')
parser.add_argument('--train-vec', type=str, default='caps_train_vecs.hdf5')
parser.add_argument('--val-data', type=str, default='caps_val.json')
parser.add_argument('--val-ids', type=str, default='caps_val_ids.json')
parser.add_argument('--val-vec', type=str, default='caps_val_vecs.hdf5')
parser.add_argument('--file-name', type=str, default='dvc01')

# Model settings
parser.add_argument('--resume-att', type=str, default='./models/att01_epoch10.pth.tar')
parser.add_argument('--resume-sg', type=str, default='./models/sg01_epoch10.pth.tar')
parser.add_argument('--resume-dvc-xe', type=str, default=None)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('-j', '--workers', type=int, default=1)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--optim', type=str, default='sgd', help='choice optimizer (adam, sgd or rms)')
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.0)

# Parameters
parser.add_argument('--feature-dim', type=int, default=500)
parser.add_argument('--num-class', type=int, default=200)
parser.add_argument('--embedding-dim', type=int, default=1024)
parser.add_argument('--hidden-dim', type=int, default=1024)
parser.add_argument('--threshold', type=float, default=0.7)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=10.0)
parser.add_argument('--alpha1', type=float, default=0.1)
parser.add_argument('--alpha2', type=float, default=0.1)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=20.0)

# Reward weights
parser.add_argument('--meteor-weight', type=float, default=1.0)
parser.add_argument('--cider-weight', type=float, default=1.0)
parser.add_argument('--bleu-weight', type=float, default=0.0)


parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-gpu', action='store_false', help='use gpu (default: True)')
parser.add_argument('--validation', action='store_true',
                    help='if True only validation else training and validation mode (default: False)')
parser.add_argument('--evaluate', action='store_false',
                    help='If True validation using ground-truth proposals mode (default: True)')
parser.add_argument('--save-every', action='store_false',
                    help='If True, save weight per every step (default: True)')
parser.add_argument('--rl-flag', action='store_true',
                    help='If True Reinforce learning else Cross entropy learning')
parser.add_argument('--gpu-devices', type=str, default='0')

# For evaluation
parser.add_argument('-r', '--references', type=str, nargs='+',
                    default=['data/captions/val_1.json', 'data/captions/val_2.json'],
                    help='reference files with ground truth captions to compare results against. delimited (,) str')
parser.add_argument('--tious', type=float,  nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                    help='Choose the tIoUs to average over.')
parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                    help='maximum propoasls per video.')

args = parser.parse_args()


class SizeError(Exception):
    def __str__(self):
        return "Batch size must be set as 1."


def main():
    global args

    if args.batch_size != 1:
        raise SizeError()

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    if not torch.cuda.is_available():
        args.use_gpu = False
    print("USE GPU: {}".format(args.use_gpu))

    # Data loader
    train_dataset = ActivityNet(args.root, args.train_data,
                                args.train_ids, args.feature_set,
                                caps=args.train_vec)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True
    )
    if args.evaluate:
        val_loader = DataLoader(
            ActivityNet(args.root, args.val_data,
                            args.val_ids, args.feature_set, mode='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers
        )
    else:
        val_loader = DataLoader(
            ActivityNet(args.root, args.val_data,
                        args.val_ids, args.feature_set,
                        caps=args.val_vec),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers
        )

    print("Train dataset : {} / Validation dataset: {}".format(
        len(train_loader.dataset), len(val_loader.dataset)))

    # Build model
    model_att, model_tep, model_sg, args.scale_ratios = build_models(
        in_c=args.feature_dim, num_class=args.num_class,
        voca_size=train_dataset.vocab_size, caps_length=train_dataset.cap_length,
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, use_gpu=args.use_gpu)
    init_eval_metric()
    if args.use_gpu:
        model_att = model_att.cuda()
        model_tep = model_tep.cuda()
        model_sg = model_sg.cuda()

    # Define loss function and optimizer
    criterion = DVCLoss(alpha=args.alpha, beta=args.beta,
                        alpha1=args.alpha1, alpha2=args.alpha2,
                        lambda1=args.lambda1, lambda2=args.lambda2,
                        use_gpu=args.use_gpu)
    params = list(model_tep.parameters()) + list(model_sg.parameters())
    if args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'rms':
        optimizer = optim.RMSprop(params, lr=args.lr)
    else:
        print("Incorrect optimizer")
        return

    # Print
    text = "\nSave file name : {}\n" \
           "Resume Attribute Detector : {}\n" \
           "Resume Sentence Generator : {}\n" \
           "Resume Dense Video Captioning with Cross Entropy Loss : {}\n" \
           "Resume Dense Video Captioning : {}\n" \
           "Reinforcement learning : {}\n" \
           "Start epoch : {}\nMax epoch : {}\n" \
           "Batch size : {}\nOptimizer : {}\n" \
           "Learning rate : {}\nMomentum : {}\nWeight decay : {}\n" \
           "Feature dimension : {}\nNum class : {}\n" \
           "Embedding dimension : {}\nHidden dimension : {}\n" \
           "Threshold : {}\nAlpha : {}\nBeta : {}\n" \
           "Alpha1 : {}\nAlpha2 : {}\nLambda1 : {}\nLambda2 : {}\n" \
           "METEOR weight : {}\nCIDEr weight : {}\nBleu@4 weight : {}\n".format(
            args.file_name, args.resume_att, args.resume_sg, args.resume_dvc_xe,
            args.resume, args.rl_flag, args.start_epoch, args.epochs,
            args.batch_size, args.optim, args.lr, args.momentum, args.weight_decay,
            args.feature_dim, args.num_class, args.embedding_dim, args.hidden_dim,
            args.threshold, args.alpha, args.beta,
            args.alpha1, args.alpha2, args.lambda1, args.lambda2,
            args.meteor_weight, args.cider_weight, args.bleu_weight
            )
    text = '='*40 + text + '='*40 + '\n'
    with open('./log/' + args.file_name + '.txt', 'w') as f:
        print(text, file=f)
    print(text)

    # Load resume from a checkpoint
    best_score = 0.0
    if args.resume_att:
        if os.path.isfile(args.resume_att):
            print("=> loading checkpoint "
                  "for attribute detector module '{}'".format(args.resume_att))
            checkpoint = torch.load(args.resume_att)
            model_att.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_att))
    if args.resume_sg:
        if os.path.isfile(args.resume_sg):
            print("=> loading checkpoint "
                  "for sentence generation module '{}'".format(args.resume_sg))
            checkpoint = torch.load(args.resume_sg)
            model_sg.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_sg))
    if args.resume_dvc_xe:
        if os.path.isfile(args.resume_dvc_xe):
            print("=> loading checkpoint "
                  "for DVC with Cross Entropy module '{}'".format(args.resume_dvc_xe))
            checkpoint = torch.load(args.resume_dvc_xe)
            model_tep.load_state_dict(checkpoint['tep_state_dict'])
            model_sg.load_state_dict(checkpoint['sg_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_dvc_xe))
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint "
                  "for DVC module '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            model_tep.load_state_dict(checkpoint['tep_state_dict'])
            model_sg.load_state_dict(checkpoint['sg_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'\n"
                  "\t : epoch {}, best score {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_score']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.validation:
        if args.evaluate:
            _ = evaluate_gt_proposal(val_loader, model_att, model_tep, model_sg, train_dataset.idx_to_word)
        else:
            _ = validate(val_loader, model_att, model_tep, model_sg)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch:", epoch)

        # train for one epoch
        train_avg_loss, train_avg_loss_event, train_avg_loss_tcr, \
            train_avg_loss_des, train_avg_loss_self_reward = train(
                train_loader, model_att, model_tep, model_sg, criterion, optimizer, epoch)

        # validation for one epoch
        if args.evaluate:
            scores = evaluate_gt_proposal(val_loader, model_att, model_tep, model_sg,
                                          train_dataset.idx_to_word, epoch+1)
            score = scores['METEOR']
        else:
            score = validate(val_loader, model_att, model_tep, model_sg)

        # remember best acc and save checkpoint
        is_best = score > best_score
        best_score = max(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'tep_state_dict': model_tep.state_dict(),
            'sg_state_dict': model_sg.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch+1, filename=args.file_name, save_every=args.save_every)

        # log
        text = "{:04d} Epoch : Train loss ({:.4f}), " \
               "Validation score ({:.4f})\n".format(
                epoch+1, train_avg_loss, score)
        with open('./log/' + args.file_name + '.txt', 'a') as f:
            print(text, file=f)


def train(train_loader, model_att, model_tep, model_sg, criterion, optimizer, epoch):
    # Train mode
    model_att.eval()
    model_tep.train()
    model_sg.train()

    losses = 0.0
    losses_event = 0.0
    losses_tcr = 0.0
    losses_des = 0.0
    losses_self_reward = 0.0
    end = time.time()
    for batch_idx, (data, boxes, sents) in enumerate(train_loader):
        if data.shape[2] < 5:
            print("Pass this data (idx:%d) because very short video." % batch_idx)
            continue
        if args.use_gpu:
            data = data.cuda()
            boxes = boxes.cuda()
            sents = sents.cuda()
        data = Variable(data)
        boxes = Variable(boxes)
        sents = Variable(sents)

        # Predict proposals
        proposals = model_tep(data)

        # Obtain positive proposal
        pos_feats, pos_sents, t_box, matches, events, pos_ids = clip_des_score(
            data, proposals[1], proposals[3], boxes, sents,
            scale_ratios=args.scale_ratios,
            threshold=args.threshold, use_gpu=args.use_gpu)

        # Generate sentences
        if args.use_gpu:
            pos_feats = pos_feats.cuda()
        pos_feats = Variable(pos_feats)
        with torch.no_grad():
            att = model_att(pos_feats)
        att = Variable(att)

        if args.rl_flag:
            gen_result, sample_logprobs = model_sg.sample(pos_feats, att, greedy=False)

            # Compute self-critical reward and sentence reward
            sent_reward, self_reward = get_sc_reward(model_sg, pos_feats, pos_sents[:, 1:],
                                                     att, gen_result,
                                                     meteor_weight=args.meteor_weight,
                                                     cider_weight=args.cider_weight,
                                                     bleu_weight=args.bleu_weight)

            sent_reward = expand_reward(t_box.shape[0], pos_ids, sent_reward)
            sent_reward = torch.from_numpy(sent_reward).float()
            self_reward = torch.from_numpy(self_reward).float()
            if args.use_gpu:
                sent_reward = sent_reward.cuda()
                self_reward = self_reward.cuda()

            # Compute loss
            loss, loss_event, loss_tcr, loss_des, loss_self_reward = criterion(
                proposals, t_box, matches, events, sent_reward,
                sample_logprobs=sample_logprobs, gen_result=gen_result.data,
                self_reward=self_reward, rl_flag=args.rl_flag)
        else:
            gen_result, output = model_sg(pos_feats, att, pos_sents[:, :-1])
            sent_reward = expand_reward(t_box.shape[0], pos_ids)
            sent_reward = torch.from_numpy(sent_reward).float()
            if args.use_gpu:
                sent_reward = sent_reward.cuda()

            # Compute loss
            loss, loss_event, loss_tcr, loss_des, loss_self_reward = criterion(
                proposals, t_box, matches, events, sent_reward,
                gen_preds=output.view(-1, output.shape[2]),
                gt_sent=pos_sents[:, 1:].contiguous().view(-1), rl_flag=args.rl_flag)

        losses += loss.item()
        losses_event += loss_event.item()
        losses_tcr += loss_tcr.item()
        losses_des += loss_des.item()
        losses_self_reward += loss_self_reward.item()

        # Compute Gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print
        if (batch_idx+1) % args.print_freq == 0:
            print("\tTrain Epoch: {} [{}/{}]\t"
                  "Time: {:.3f}\t"
                  "Loss: {:.6f}".format(
                   epoch, (batch_idx+1), len(train_loader), time.time()-end, losses))

    avg_loss = losses/len(train_loader.dataset)
    print("Epoch {} average loss : {:.6f}".format(epoch, avg_loss))
    avg_loss_event = losses_event/len(train_loader.dataset)
    avg_loss_tcr = losses_tcr/len(train_loader.dataset)
    avg_loss_des = losses_des/len(train_loader.dataset)
    avg_loss_self_reward = losses_self_reward/len(train_loader.dataset)

    return avg_loss, avg_loss_event, avg_loss_tcr, avg_loss_des, avg_loss_self_reward


def validate(val_loader, model_att, model_tep, model_sg):
    # Evaluate mode
    model_att.eval()
    model_tep.eval()
    model_sg.eval()

    sum_score = 0.0
    num_val = 0
    end = time.time()
    with torch.no_grad():
        for batch_idx, (data, boxes, sents) in enumerate(val_loader):
            if data.shape[2] < 5:
                print("Pass this data (idx:%d) because very short video." % batch_idx)
                continue
            if args.use_gpu:
                data = data.cuda()
                boxes = boxes.cuda()
                sents = sents.cuda()
            data = Variable(data)
            boxes = Variable(boxes)
            sents = Variable(sents)

            # Predict proposals
            proposals = model_tep(data)

            # Obtain positive proposal
            pos_feats, pos_sents, _, _, _, _ = clip_des_score(
                data, proposals[1], proposals[3], boxes, sents,
                scale_ratios=args.scale_ratios,
                threshold=args.threshold, use_gpu=args.use_gpu)

            # Generate sentences
            if args.use_gpu:
                pos_feats = pos_feats.cuda()
            pos_feats = Variable(pos_feats)
            att = model_att(pos_feats)
            att = Variable(att)
            gen_result, _ = model_sg.sample(pos_feats, att, greedy=True)

            # Measure
            scores = compute_meteor_score(gen_result, pos_sents[:, 1:])
            sum_score += sum(scores)
            num_val += pos_sents.shape[0]

            # Print
            if (batch_idx + 1) % args.print_freq == 0:
                print('\tValidation: [{}/{}]\t'
                      'Time: {:.3f}\t'
                      'Score: {:.3f}'.format(
                        (batch_idx + 1), len(val_loader), time.time() - end, sum(scores)))

    avg_score = sum_score/float(num_val)
    print("Validation average score: {:.4f}".format(avg_score))

    return avg_score


def evaluate_gt_proposal(val_loader, model_att, model_tep, model_sg, idx_to_word, epoch=0):
    # Evaluate mode
    model_att.eval()
    model_tep.eval()
    model_sg.eval()

    out = {}
    out['version'] = 'VERSION 1.0'
    out['results'] = {}
    out['external_data'] = {}
    out['external_data']['used'] = 'false'
    out['external_data']['details'] = 'for evaluation'
    end = time.time()

    with torch.no_grad():
        for batch_idx, (data, boxes, duration, v_name, timestamp) in enumerate(val_loader):
            if data.shape[2] < 5:
                print("Pass this data (idx:%d) because very short video." % batch_idx)
                continue
            if args.use_gpu:
                data = Variable(data.cuda())
                boxes = Variable(boxes.cuda())
            else:
                data = Variable(data)
                boxes = Variable(boxes)

            # Predict proposals
            proposals = model_tep(data)

            # Obtain proposal features with ground-truth proposal
            # using weighted attention(descriptiveness) score
            pos_feats = get_gt_proposal(
                data, proposals[1], proposals[3], boxes,
                scale_ratios=args.scale_ratios, use_gpu=args.use_gpu)

            if args.use_gpu:
                pos_feats = Variable(pos_feats.cuda())
            else:
                pos_feats = Variable(pos_feats)
            att = model_att(pos_feats)
            if args.use_gpu:
                att = Variable(att.cuda())
            else:
                att = Variable(att)
            att = Variable(att)

            # Generate sentences
            gen_result, _ = model_sg.sample(pos_feats, att, greedy=True)
            gen_sents = idx_to_sent(gen_result, idx_to_word)

            start_times = timestamp[0, :, 0].data.cpu().numpy()
            end_times = timestamp[0, :, 1].data.cpu().numpy()

            out['results'][v_name[0]] = []
            for i in range(len(gen_sents)):
                temp = {}
                temp['sentence'] = gen_sents[i][0]
                temp['timestamp'] = [float(start_times[i]), float(end_times[i])]
                out['results'][v_name[0]].append(temp)

            # Print
            if (batch_idx + 1) % args.print_freq == 0:
                print("\tValidation: [{}/{}]\t"
                      "Time: {:.3f}".format(
                        (batch_idx + 1), len(val_loader), time.time() - end))

    # Write to JSON
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    json_name = 'output/result_{}_{}.json'.format(args.file_name, str(epoch))
    json.dump(out, open(json_name, 'w'))

    # Evaluate scores
    scores = {}
    evaluator = ANETcaptions(ground_truth_filenames=args.references,
                             prediction_filename=json_name,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             verbose=True)
    evaluator.evaluate()
    print("Validation Scores")
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        scores[metric] = 100 * sum(score) / float(len(score))
        print('| %s: %2.4f' % (metric, scores[metric]))

    return scores


def save_checkpoint(state, is_best, epoch, filename='checkpoint', save_every=False):
    if not os.path.isdir('./models'):
        os.makedirs('./models')
    if save_every:
        torch.save(state, './models/{}_epoch{}.pth.tar'.format(filename, epoch))
    else:
        torch.save(state, './models/{}.pth.tar'.format(filename))
        if is_best:
            shutil.copyfile('./models/{}.pth.tar'.format(filename),
                            './models/{}_best.pth.tar'.format(filename))


if __name__ == '__main__':
    main()
