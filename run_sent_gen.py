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
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_manager import ActivityNetSG
from model import build_sg_model
from rewards import init_eval_metric, compute_meteor_score
from utils import idx_to_sent

sys.path.insert(0, './densevid_eval')
from evaluate import ANETcaptions

parser = argparse.ArgumentParser(description='Sentence Generation')

# Data input settings
parser.add_argument('--root', type=str, default='data/actnet')
parser.add_argument('--train-file', type=str, default='sg_train.hdf5')
parser.add_argument('--val-file', type=str, default='sg_val.hdf5')
parser.add_argument('--val-gt-props', type=str, default='sg_val_eval.json')
parser.add_argument('--word-to-idx', type=str, default='sg_word_to_idx.json')
parser.add_argument('--file-name', type=str, default='sg01')

# Model settings
parser.add_argument('--resume-att', type=str, default='./models/att01_epoch10.pth.tar')
parser.add_argument('--resume-sg', type=str, default=None)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--optim', type=str, default='adam', help='choice optimizer (adam or sgd)')
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.0)

# Parameters
parser.add_argument('--feature-dim', type=int, default=500)
parser.add_argument('--num-class', type=int, default=200)
parser.add_argument('--embedding-dim', type=int, default=1024)
parser.add_argument('--hidden-dim', type=int, default=1024)

parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-gpu', action='store_false', help='use gpu (default: True)')
parser.add_argument('--validation', action='store_true',
                    help='if True only validation else training and validation mode (default: False)')
parser.add_argument('--evaluate', action='store_false',
                    help='If True validation using ground-truth proposals mode (default: True)')
parser.add_argument('--save-every', action='store_false',
                    help='If True, save weight per every step (default: True)')
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


def main():
    global args

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    if not torch.cuda.is_available():
        args.use_gpu = False
    print("USE GPU: {}".format(args.use_gpu))

    # Data loader
    train_dataset = ActivityNetSG(args.root, args.train_file, word_to_idx=args.word_to_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    if args.evaluate:
        val_loader = DataLoader(
            ActivityNetSG(args.root, args.val_file, gt_props=args.val_gt_props, mode='eval'),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    else:
        val_loader = DataLoader(
            ActivityNetSG(args.root, args.val_file),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    print("Train dataset : {} / Validation dataset: {}".format(
        len(train_loader.dataset), len(val_loader.dataset)))

    # Build model
    model_att, model_sg = build_sg_model(in_c=args.feature_dim,
                                         num_class=args.num_class,
                                         voca_size=train_dataset.vocab_size,
                                         caps_length=train_dataset.cap_length,
                                         embedding_dim=args.embedding_dim,
                                         hidden_dim=args.hidden_dim,
                                         use_gpu=args.use_gpu)
    init_eval_metric()
    if args.use_gpu:
        model_att = model_att.cuda()
        model_sg = model_sg.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    if args.optim == 'adam':
        optimizer = optim.Adam(model_sg.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model_sg.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Incorrect optimizer.")
        return
    if args.use_gpu:
        criterion = criterion.cuda()

    # Print
    text = "\nSave file name : {}\n" \
           "Resume Attribute Detector : {}\n" \
           "Resume Sentence Generator : {}\n" \
           "Start epoch : {}\nMax epoch : {}\n" \
           "Batch size : {}\nOptimizer : {}\n" \
           "Learning rate : {}\nMomentum : {}\nWeight decay : {}\n" \
           "Feature dimension : {}\nNum class : {}\n" \
           "Embedding dimension : {}\nHidden dimension : {}\n".format(
            args.file_name, args.resume_att, args.resume_sg, args.start_epoch, args.epochs,
            args.batch_size, args.optim, args.lr, args.momentum, args.weight_decay,
            args.feature_dim, args.num_class, args.embedding_dim, args.hidden_dim
            )
    text = '='*40 + text + '='*40 + '\n'
    if not os.path.isdir('./log'):
        os.makedirs('./log')
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
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            model_sg.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'\n"
                  "\t : epoch {}, best score {}"
                  .format(args.resume_sg, checkpoint['epoch'], checkpoint['best_score']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_sg))

    # CUDNN benchmark is look for the optimal set of algorithms for
    # that particular configuration (which takes some time).
    # This usually leads to faster runtime.
    # But if your input sizes changes at each iteration, leads to worse runtime.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # cudnn.benchmark = True

    if args.validation:
        if args.evaluate:
            _, _ = evaluate_gt(val_loader, model_att, model_sg, criterion, train_dataset.idx_to_word)
        else:
            _, _ = validate(val_loader, model_att, model_sg, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch:", epoch)

        # train for one epoch
        train_avg_loss = train(train_loader, model_att, model_sg, criterion, optimizer, epoch)

        # validation for one epoch
        if args.evaluate:
            val_avg_loss, scores = evaluate_gt(val_loader, model_att, model_sg, criterion,
                                               train_dataset.idx_to_word, epoch+1)
            score = scores['METEOR']
        else:
            val_avg_loss, score = validate(val_loader, model_att, model_sg, criterion)

        # remember best acc and save checkpoint
        is_best = score > best_score
        best_score = max(score, best_score)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_sg.state_dict(),
            'best_score': best_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch+1, filename=args.file_name, save_every=args.save_every)

        # log
        text = "{:04d} Epoch : Train loss ({:.4f}), " \
               "Validation score ({:.4f})\n".format(
                epoch+1, train_avg_loss, score)
        with open('./log/' + args.file_name + '.txt', 'a') as f:
            print(text, file=f)


def train(train_loader, model_att, model_sg, criterion, optimizer, epoch):
    # Train mode
    model_att.eval()
    model_sg.train()

    losses = 0.0
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.use_gpu:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        target = Variable(target)

        # Generate sentences
        with torch.no_grad():
            att = model_att(data)
        att = Variable(att)
        _, output = model_sg(data, att, target[:, :-1])

        # Compute loss
        loss = criterion(output.view(-1, output.shape[2]),
                         target[:, 1:].contiguous().view(-1))

        losses += loss.item()

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

    return avg_loss


def validate(val_loader, model_att, model_sg, criterion):
    # Evaluate mode
    model_att.eval()
    model_sg.eval()

    losses = 0.0
    sum_score = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if args.use_gpu:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data)
            target = Variable(target)

            # Generate sentences
            att = model_att(data)
            att = Variable(att)
            gen_result, output = model_sg(data, att)
            loss = criterion(output.view(-1, output.shape[2]),
                             target[:, 1:].contiguous().view(-1))

            losses += loss.item()

            # Measure
            scores = compute_meteor_score(gen_result, target[:, 1:])
            sum_score += sum(scores)

    avg_loss = losses / len(val_loader.dataset)
    avg_score = sum_score/len(val_loader.dataset)
    print("Validation average loss : {:.4f} , score: {:.4f}".format(avg_loss, avg_score))

    return avg_loss, avg_score


def evaluate_gt(val_loader, model_att, model_sg, criterion, idx_to_word, epoch=0):
    # Evaluate mode
    model_att.eval()
    model_sg.eval()

    count = 0
    losses = 0.0
    out2 = {}
    out2['version'] = 'VERSION 1.0'
    out2['results'] = {}
    out2['external_data'] = {}
    out2['external_data']['used'] = 'false'
    out2['external_data']['details'] = 'for evaluation'

    with torch.no_grad():
        for batch_idx, (data, target, v_name, timestamp) in enumerate(val_loader):
            if args.use_gpu:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data)
            target = Variable(target)

            # Generate sentences
            att = model_att(data)
            att = Variable(att)
            gen_result, output = model_sg(data, att)
            loss = criterion(output.view(-1, output.shape[2]),
                             target[:, 1:].contiguous().view(-1))

            losses += loss.item()

            # Into dict. structure
            gen_sents = idx_to_sent(gen_result, idx_to_word)
            start_times = timestamp[:, 0].data.cpu().numpy()
            end_times = timestamp[:, 1].data.cpu().numpy()

            for i in range(len(gen_sents)):
                if not v_name[i] in out2['results']:
                    out2['results'][v_name[i]] = []
                temp = {}
                temp['sentence'] = gen_sents[i][0]
                temp['timestamp'] = [float(start_times[i]), float(end_times[i])]
                out2['results'][v_name[i]].append(temp)
                count += 1

    print("Check Validation data : {} / {}".format(count, len(val_loader.dataset)))

    avg_loss = losses / len(val_loader.dataset)
    print("Validation average loss : {:.4f}".format(avg_loss))

    # Write to JSON
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    json_name = 'output/result_{}_{}.json'.format(args.file_name, str(epoch))
    json.dump(out2, open(json_name, 'w'))

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

    return avg_loss, scores


def save_checkpoint(state, is_best, epoch, filename='checkpoint', save_every=False):
    if save_every:
        torch.save(state, '{}_epoch{}.pth.tar'.format(filename, epoch))
    else:
        torch.save(state, '{}.pth.tar'.format(filename))
        if is_best:
            shutil.copyfile('{}.pth.tar'.format(filename), '{}_best.pth.tar'.format(filename))


if __name__ == '__main__':
    main()
