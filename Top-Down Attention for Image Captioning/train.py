from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.load_data import LoadData

import sys
import argparse
import tensorflow as tf


def main(params):
    loader = LoadData(input_json=params.input_labels, input_h5=params.input_caps, feature_path=params.input_feats,
                      batch_img=params.batch_img, seq_per_img=params.seq_per_img, train_only=params.train_only)

    if params.mode == "EVAL":
        model = CaptionGenerator(loader.word_to_idx, num_features=params.num_objects, dim_feature=params.dim_features,
                                 dim_embed=params.dim_word_emb, dim_hidden=params.rnn_hid_size,
                                 dim_attention=params.att_hid_size, n_time_step=loader.seq_length-1)
        solver = CaptioningSolver(data_loader=loader, model=model, ngram_file=params.input_ngram,
                                  n_epochs=params.epoch, update_rule=params.optimizer, learning_rate=params.lr,
                                  print_every=params.print_every, start_epoch=params.start_epoch,
                                  log_path=params.log_path, model_path=params.model_path,
                                  pretrained_model=params.pretrained, test_model=params.test_model)
        solver.output_result_for_eval()
    elif params.mode == "TEST":
        model = CaptionGenerator(loader.word_to_idx, num_features=params.num_objects, dim_feature=params.dim_features,
                                 dim_embed=params.dim_word_emb, dim_hidden=params.rnn_hid_size,
                                 dim_attention=params.att_hid_size, n_time_step=loader.seq_length-1)
        solver = CaptioningSolver(data_loader=loader, model=model, ngram_file=params.input_ngram,
                                  n_epochs=params.epoch, update_rule=params.optimizer, learning_rate=params.lr,
                                  print_every=params.print_every, start_epoch=params.start_epoch,
                                  log_path=params.log_path, model_path=params.model_path,
                                  pretrained_model=params.pretrained, test_model=params.test_model)
        solver.test()
    elif params.mode == "TRAIN":
        with tf.Graph().as_default():
            model = CaptionGenerator(loader.word_to_idx, num_features=params.num_objects,
                                     dim_feature=params.dim_features, dim_embed=params.dim_word_emb,
                                     dim_hidden=params.rnn_hid_size, dim_attention=params.att_hid_size,
                                     n_time_step=loader.seq_length-1)
            solver = CaptioningSolver(data_loader=loader, model=model, ngram_file=params.input_ngram,
                                      n_epochs=params.epoch, update_rule=params.optimizer, learning_rate=params.lr,
                                      print_every=params.print_every, start_epoch=params.start_epoch,
                                      log_path=params.log_path, model_path=params.model_path,
                                      pretrained_model=params.pretrained, test_model=params.test_model)
            solver.train()
        with tf.Graph().as_default():
            model = CaptionGenerator(loader.word_to_idx, num_features=params.num_objects,
                                     dim_feature=params.dim_features, dim_embed=params.dim_word_emb,
                                     dim_hidden=params.rnn_hid_size, dim_attention=params.att_hid_size,
                                     n_time_step=loader.seq_length-1)
            solver = CaptioningSolver(data_loader=loader, model=model, ngram_file=params.input_ngram,
                                      n_epochs=params.epoch, update_rule=params.optimizer, learning_rate=params.lr,
                                      print_every=params.print_every, start_epoch=params.start_epoch,
                                      log_path=params.log_path, model_path=params.model_path,
                                      pretrained_model=params.pretrained, test_model=params.test_model)
            solver.train_reinforce()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # Data inputs
    parser.add_argument('--input_labels', default='data/coco/coco_labels.json',
                        help='Input labels data file (generate from prepro_captions.py)')
    parser.add_argument('--input_caps', default='data/coco/coco_captions.h5',
                        help='Input captions data file (generate from prepro_captions.py)')
    parser.add_argument('--input_feats', default='data/features',
                        help='Input features data directory (generate from prepro_features.py)')
    parser.add_argument('--input_ngram', default='data/coco/coco-train-idxs.p',
                        help='Input ngram data file (generate from prepro_ngram.py)')

    # Informations for data
    parser.add_argument('--train_only', default=0, type=int,
                        help='if 1 then only use train set in COCO for training, else use train and restval set')
    parser.add_argument('--num_objects', default=36, type=int, help='number of objects(features) in each image for CNN')
    parser.add_argument('--dim_features', default=2048, type=int, help='dimension of the image feature')

    # Parameters for network
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start number epoch. if you first train model then is 0')
    parser.add_argument('--batch_img', default=20, type=int, help='number of images for each minibatch')
    parser.add_argument('--seq_per_img', default=5, type=int, help='number of captions for each image')
    parser.add_argument('--optimizer', default='momentum', help='Optimizer')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--dim_word_emb', default=1000, type=int,
                        help='dimension for the word embedding in the caption model')
    parser.add_argument('--att_hid_size', default=512, type=int, help='size of the hidden unit in the attention layer')
    parser.add_argument('--rnn_hid_size', default=1000, type=int, help='size of the hidden unit in the LSTM')

    parser.add_argument('--mode', default='EVAL', help='select mode [TRAIN, TEST, EVAL]')
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--log_path', default='log/')
    parser.add_argument('--model_path', default='model/')
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--test_model', default='./model/model_rl-10')

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
