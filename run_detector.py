from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import shutil
import argparse

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_manager import ActivityNetAtts
from model import build_att_model


parser = argparse.ArgumentParser(description='Attribute Classification (Detector)')

# Data input settings
parser.add_argument('--root', type=str, default='data/actnet')
parser.add_argument('--train-file', type=str, default='att_train.hdf5')
parser.add_argument('--val-file', type=str, default='att_val.hdf5')
parser.add_argument('--file-name', type=str, default='att01')

# Model settings
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.0001)

# Parameters
parser.add_argument('--feature-dim', type=int, default=500)
parser.add_argument('--num-class', type=int, default=200)

parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-gpu', action='store_false', help='use gpu (default: True)')
parser.add_argument('--validation', action='store_true',
                    help='if True only validation else training and validation mode (default: False)')
parser.add_argument('--save-every', action='store_false',
                    help='If True, save weight per every step (default: True)')
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
    train_dataset = ActivityNetAtts(args.root, args.train_file)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        ActivityNetAtts(args.root, args.val_file),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    print("Train dataset : {} / Validation dataset: {}".format(
        len(train_loader.dataset), len(val_loader.dataset)))

    # Build model
    model = build_att_model(in_c=args.feature_dim, num_class=args.num_class)
    if args.use_gpu:
        model = model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.use_gpu:
        criterion = criterion.cuda()

    # Print
    text = "\nSave file name : {}\n" \
           "Resume Attribute Detector : {}\n" \
           "Start epoch : {}\nMax epoch : {}\n" \
           "Batch size : {}\nLearning rate : {}\n" \
           "Momentum : {}\nWeight decay : {}\n" \
           "Feature dimension : {}\nNum class : {}\n".format(
            args.file_name, args.resume, args.start_epoch, args.epochs,
            args.batch_size, args.lr, args.momentum, args.weight_decay,
            args.feature_dim, args.num_class
            )
    text = '='*40 + text + '='*40 + '\n'
    if not os.path.isdir('./log'):
        os.makedirs('./log')
    with open('./log/' + args.file_name + '.txt', 'w') as f:
        print(text, file=f)
    print(text)

    # Load resume from a checkpoint
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'\n"
                  "\t : epoch {}, best_accuracy {}"
                  .format(args.resume, checkpoint['epoch'], checkpoint['best_acc']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # CUDNN benchmark is look for the optimal set of algorithms for
    # that particular configuration (which takes some time).
    # This usually leads to faster runtime.
    # But if your input sizes changes at each iteration, leads to worse runtime.
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    cudnn.benchmark = True

    if args.validation:
        _ = validate(val_loader, model)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch:", epoch)

        # train for one epoch
        train_avg_loss = train(train_loader, model, criterion, optimizer, epoch)

        # validation for one epoch
        acc = validate(val_loader, model)

        # update best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch+1, filename=args.file_name, save_every=args.save_every)

        # log
        text = "{:04d} Epoch : Train loss ({:.4f}), " \
               "Validation accuracy ({:.4f})\n".format(
                epoch+1, train_avg_loss, acc)
        with open('./log/' + args.file_name + '.txt', 'a') as f:
            print(text, file=f)


def train(train_loader, model, criterion, optimizer, epoch):
    # Train mode
    model.train()

    losses = 0.0
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.use_gpu:
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        target = Variable(target)

        # Predict attribute
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        losses += loss.item()

        # Compute Gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print
        if (batch_idx+1) % args.print_freq == 0:
            print('\tTrain Epoch: {} [{}/{}]\t'
                  'Time: {:.3f}\t'
                  'Loss: {:.6f}'.format(
                   epoch, (batch_idx+1), len(train_loader), time.time()-end, losses))

    avg_loss = losses/len(train_loader.dataset)
    print('Epoch {} average loss : {:.6f}'.format(epoch, avg_loss))

    return avg_loss


def validate(val_loader, model):
    # Evaluate mode
    model.eval()

    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            if args.use_gpu:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data)
            target = Variable(target)

            # Predict attribute
            output = model(data)

            # Measure
            predict = torch.argmax(output, dim=1)
            correct = predict.eq(target)
            num_correct += correct.sum().item()

        acc = float(num_correct)/len(val_loader.dataset)
        print('Validation Acc: {:.4f}'.format(acc))

    return acc


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
