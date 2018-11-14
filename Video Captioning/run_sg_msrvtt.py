from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import argparse
import json

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_manager_msrvtt import MSRVTTSG, MSRVTTSGRL, MSRVTTSGEval
from loss import SelfCritLoss
from model import build_sg_model
from eval_msrvtt import evaluate
from rewards import init_eval_metric, get_sc_reward_msrvtt
from utils import idx_to_sent

parser = argparse.ArgumentParser(description='Sentence Generation')

# Data input settings
parser.add_argument('--root', type=str, default='data/MSRVTT')
parser.add_argument('--train-file', type=str, default='sg_train.hdf5')
parser.add_argument('--train-rl-file', type=str, default='sg_train_rl.hdf5')
parser.add_argument('--train-vid', type=str, default='sg_train_vid.json')
parser.add_argument('--train-eval', type=str, default='sg_train_eval.json')
parser.add_argument('--val-file', type=str, default='sg_val.hdf5')
parser.add_argument('--val-vid', type=str, default='sg_val_vid.json')
parser.add_argument('--val-eval', type=str, default='sg_val_eval.json')
parser.add_argument('--word-to-idx', type=str, default='sg_word_to_idx.json')
parser.add_argument('--file-name', type=str, default='msrvtt_sg01')

# Model settings
parser.add_argument('--resume-att', type=str, default=None)
parser.add_argument('--resume-sg', type=str, default=None)
parser.add_argument('--resume-sg-rl', type=str, default=None)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--optim', type=str, default='adam', help='choice optimizer (adam, rms or sgd)')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.0)

# Parameters
parser.add_argument('--feature-dim', type=int, default=4096)
parser.add_argument('--num-class', type=int, default=20)
parser.add_argument('--embedding-dim', type=int, default=1024)
parser.add_argument('--hidden-dim', type=int, default=1024)

# Reward weights
parser.add_argument('--meteor-weight', type=float, default=0.0)
parser.add_argument('--cider-weight', type=float, default=0.0)
parser.add_argument('--bleu-weight', type=float, default=1.0)
parser.add_argument('--bleu-ngram', type=int, default=2)

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

args = parser.parse_args()


def main():
    global args

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    if not torch.cuda.is_available():
        args.use_gpu = False
    print("USE GPU: {}".format(args.use_gpu))

    # Data loader
    if args.rl_flag:
        train_dataset = MSRVTTSGRL(args.root, args.train_rl_file, args.train_vid, args.word_to_idx)
    else:
        train_dataset = MSRVTTSG(args.root, args.train_file, args.word_to_idx)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    if args.evaluate:
        val_loader = DataLoader(
            MSRVTTSGEval(args.root, args.val_file, gt_props=args.val_vid),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
    else:
        val_loader = DataLoader(
            MSRVTTSG(args.root, args.val_file),
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
    if args.rl_flag:
        init_eval_metric(bleu_n=args.bleu_ngram)
    train_gts = build_ground_truths(os.path.join(args.root, args.train_eval))
    val_gts = build_ground_truths(os.path.join(args.root, args.val_eval))
    if args.use_gpu:
        model_att = model_att.cuda()
        model_sg = model_sg.cuda()

    # Define loss function and optimizer
    if args.rl_flag:
        criterion = SelfCritLoss()
    else:
        criterion = nn.CrossEntropyLoss(size_average=False)
    if args.optim == 'adam':
        optimizer = optim.Adam(model_sg.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model_sg.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'rms':
        optimizer = optim.RMSprop(model_sg.parameters(), lr=args.lr)
    else:
        print("Incorrect optimizer.")
        return
    if args.use_gpu:
        criterion = criterion.cuda()

    # Print
    text = "\nSave file name : {}\n" \
           "Resume Attribute Detector : {}\n" \
           "Resume Sentence Generator : {}\n" \
           "Resume Sentence Generator(RL) : {}\n" \
           "Reinforcement learning : {}\n" \
           "Start epoch : {}\nMax epoch : {}\n" \
           "Batch size : {}\nOptimizer : {}\n" \
           "Learning rate : {}\nMomentum : {}\nWeight decay : {}\n" \
           "Feature dimension : {}\nNum class : {}\n" \
           "Embedding dimension : {}\nHidden dimension : {}\n" \
           "METEOR weight : {}\nCIDEr weight : {}\nBleu@N weight : {}\nBleu@N-{}\n".format(
            args.file_name, args.resume_att, args.resume_sg, args.resume_sg_rl,
            args.rl_flag, args.start_epoch, args.epochs,
            args.batch_size, args.optim, args.lr, args.momentum, args.weight_decay,
            args.feature_dim, args.num_class, args.embedding_dim, args.hidden_dim,
            args.meteor_weight, args.cider_weight, args.bleu_weight, args.bleu_ngram
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
            model_sg.load_state_dict(checkpoint['state_dict'])
            if not args.rl_flag:
                args.start_epoch = checkpoint['epoch']
                best_score = checkpoint['best_score']
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}'\n"
                      "\t : epoch {}, best score {}"
                      .format(args.resume_sg, checkpoint['epoch'], checkpoint['best_score']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_sg))
    if args.resume_sg_rl:
        if os.path.isfile(args.resume_sg_rl):
            print("=> loading checkpoint "
                  "for sentence generation module '{}'".format(args.resume_sg_rl))
            checkpoint = torch.load(args.resume_sg_rl)
            model_sg.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'\n"
                  "\t : epoch {}, best score {}"
                  .format(args.resume_sg_rl, checkpoint['epoch'], checkpoint['best_score']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_sg_rl))

    # CUDNN benchmark is look for the optimal set of algorithms for
    # that particular configuration (which takes some time).
    # This usually leads to faster runtime.
    # But if your input sizes changes at each iteration, leads to worse runtime.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # cudnn.benchmark = True

    if args.validation:
        if args.evaluate:
            _, _ = evaluate_gt(val_loader, model_att, model_sg, criterion, train_dataset.idx_to_word, val_gts)
        else:
            _, _ = validate(val_loader, model_att, model_sg, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch:", epoch)

        # train for one epoch
        if args.rl_flag:
            train_avg_loss = train_rl(train_loader, model_att, model_sg, criterion, optimizer,
                                      train_gts, train_dataset.idx_to_word, epoch)
        else:
            train_avg_loss = train(train_loader, model_att, model_sg, criterion, optimizer, epoch)

        # validation for one epoch
        if args.evaluate:
            val_avg_loss, scores = evaluate_gt(val_loader, model_att, model_sg, criterion,
                                               train_dataset.idx_to_word, val_gts, epoch+1)
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


def train_rl(train_loader, model_att, model_sg, criterion, optimizer, gts, idx_to_word, epoch):
    # Train mode
    model_att.eval()
    model_sg.train()

    losses = 0.0
    end = time.time()
    for batch_idx, (data, v_name) in enumerate(train_loader):
        if args.use_gpu:
            data = data.cuda()
        data = Variable(data)

        # Generate sentences
        with torch.no_grad():
            att = model_att(data)
        att = Variable(att)
        gen_result, sample_logprobs = model_sg.sample(data, att, greedy=False)
        self_reward = get_sc_reward_msrvtt(model_sg, data, att, gen_result,
                                    v_name, gts, idx_to_word,
                                    meteor_weight=args.meteor_weight,
                                    cider_weight=args.cider_weight,
                                    bleu_weight=args.bleu_weight)
        self_reward = torch.from_numpy(self_reward).float()
        if args.use_gpu:
            self_reward = self_reward.cuda()

        # Compute loss
        loss = criterion(sample_logprobs=sample_logprobs, gen_result=gen_result.data,
                         self_reward=self_reward)
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
    score = 0.0
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

    avg_loss = losses / len(val_loader.dataset)
    print("Validation average loss : {:.4f} , score: {:.4f}".format(avg_loss, score))

    return avg_loss, score


def evaluate_gt(val_loader, model_att, model_sg, criterion, idx_to_word, gts, epoch=0):
    # Evaluate mode
    model_att.eval()
    model_sg.eval()

    count = 0
    losses = 0.0
    predictions = {}

    with torch.no_grad():
        for batch_idx, (data, target, v_name) in enumerate(val_loader):
            if args.use_gpu:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data)
            target = Variable(target)

            # Generate sentences
            att = model_att(data)
            att = Variable(att)
            gen_result, output = model_sg(data, att)
            if not args.rl_flag:
                loss = criterion(output.view(-1, output.shape[2]),
                                 target[:, 1:].contiguous().view(-1))
                losses += loss.item()

            # Into dict. structure
            gen_sents = idx_to_sent(gen_result, idx_to_word)

            for i in range(len(gen_sents)):
                if not int(v_name[i][5:]) in predictions:
                    predictions[int(v_name[i][5:])] = []
                    temp = {}
                    temp['caption'] = gen_sents[i][0]
                    predictions[int(v_name[i][5:])].append(temp)
                count += 1

    print("Check Validation data : {} / {}".format(count, len(val_loader.dataset)))

    avg_loss = losses / len(val_loader.dataset)
    print("Validation average loss : {:.4f}".format(avg_loss))

    # Write to JSON
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    json_name = 'output/result_{}_{}.json'.format(args.file_name, str(epoch))
    json.dump(predictions, open(json_name, 'w'))

    # Evaluate scores
    scores = evaluate(gts, predictions)
    for key in scores.keys():
        scores[key] *= 100

    return avg_loss, scores


def build_ground_truths(gt_file):
    data = json.load(open(gt_file, 'r'))

    gts = {}
    for key in data.keys():
        gts[int(key[5:])] = []
        for i in range(len(data[key])):
            temp = {}
            temp['caption'] = data[key][i]['sentence']
            gts[int(key[5:])].append(temp)

    return gts


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
