import os
import time
import numpy as np
from collections import OrderedDict
import skimage
import skimage.io
import skimage.color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

from core.load_data import *
from core.utils import *
from pyciderevalcap.ciderD.ciderD import CiderD


class CaptioningSolver(object):
    def __init__(self, data_loader, model, **kwargs):
        """
        Args:
            - data_loader : Load dataset(features, captions, and informations...).
            - model: Caption generation model.
        Option:
            - num_epochs: (Integer) The number of epochs to run for training.
            - update_rule: (String) A string giving the name of an update rule.
            - learning_rate: (Float) Learning rate.
            - print_every: (Integer) Training losses will be printed every print_every iterations.
            - start_epoch: (Integer) Start number of epoch. If use pretrained model then is not 0.
            - log_path: (String) Log path for summary.
            - model_path: (String) Model path for saving.
            - pretrained_model: (String) Pretrained model path.
            - test_model: (String) Model path for test.
        """

        self.data_loader = data_loader
        self.model = model
        self.ngram_file = kwargs.pop('ngram_file', None)
        self.num_epochs = kwargs.pop('n_epochs', 10)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_every = kwargs.pop('print_every', 100)
        self.start_epoch = kwargs.pop('start_epoch', 0)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/model_rl-10')

        self.num_train_img = data_loader.num_train_images
        self.batch_size = data_loader.batch_size
        self.num_iters_per_epoch = int(np.floor(self.num_train_img / data_loader.batch_img))

        self.CiderD_scorer = None

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _init_cider_scorer(self):
        cached_tokens, _ = os.path.splitext(os.path.basename(self.ngram_file))
        self.CiderD_scorer = self.CiderD_scorer or CiderD(df=cached_tokens, ngram_file=self.ngram_file)

    def _get_self_critical_reward(self, sampled_cap, baseline_cap, data, print_flag=False):
        res = OrderedDict()
        for i in range(self.batch_size):
            res[i] = [array_to_str_for_score(sampled_cap[i])]
        for i in range(self.batch_size):
            res[self.batch_size + i] = [array_to_str_for_score(baseline_cap[i])]

        gts = OrderedDict()
        for i in range(len(data['gts'])):
            gts[i] = [array_to_str_for_score(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

        res = [{'image_id': i, 'caption': res[i]} for i in range(2 * self.batch_size)]
        gts = {i: gts[i % self.batch_size // self.data_loader.seq_per_img] for i in range(2 * self.batch_size)}
        _, scores = self.CiderD_scorer.compute_score(gts, res)

        if print_flag:
            aa = np.mean(np.array(scores[:self.batch_size]))
            np.mean(aa)
            print("Sampled cap reward : %f, Greedy cap reward : %f" %
                  (np.mean(scores[:self.batch_size]), np.mean(scores[self.batch_size:])))

        scores = scores[:self.batch_size] - scores[self.batch_size:]
        rewards = np.array(scores)

        return rewards

    def train(self):
        # build graphs for training model
        loss = self.model.build_model_xe()

        # train op
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            decay_steps = (self.num_epochs + 1) * self.num_iters_per_epoch
            lr = tf.train.polynomial_decay(self.learning_rate, global_step=global_step,
                                           decay_steps=decay_steps, end_learning_rate=0.00001, power=1.0)
            if self.update_rule == 'momentum':
                optimizer = self.optimizer(learning_rate=lr, momentum=0.9)
            else:
                optimizer = self.optimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss, global_step=global_step)

        tf.get_variable_scope().reuse_variables()
        _, generated_greedy_caps = self.model.build_sampler(max_len=self.data_loader.seq_length-1, max_out=True)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        tf.summary.scalar('learning_rate', lr)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()

        # GPU options
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        print("Cross Entropy Loss Training")
        print("The number of epoch: %d" % self.num_epochs)
        print("Data size: %d" % self.num_train_img)
        print("Batch size: %d" % self.batch_size)
        print("Iterations per epoch: %d" % self.num_iters_per_epoch)

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.start_epoch, self.num_epochs):
                # shuffle training dataset
                self.data_loader.shuffle_data()

                # training
                for i in range(self.num_iters_per_epoch):
                    data_batch = self.data_loader.get_batch('train', i)
                    features_batch = data_batch['features']
                    captions_batch = data_batch['labels']
                    feed_dict = {self.model.features: features_batch,
                                 self.model.captions: captions_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e * self.num_iters_per_epoch + i)

                    if (i + 1) % self.print_every == 0:
                        print("\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f"
                              % (e + 1, i + 1, l))
                        ground_truths = data_batch['gts'][0]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print("Ground truth %d: %s" % (j + 1, gt))
                        gen_caps = sess.run(generated_greedy_caps, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print("Generated caption: %s\n" % decoded[0])

                print("Previous epoch loss: ", prev_loss)
                print("Current epoch loss: ", curr_loss)
                print("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

                # save model's parameters
                saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                print("model-%s saved." % (e + 1))

    def train_reinforce(self):
        # build graphs for training model
        loss = self.model.build_model_reinforce()

        # train op
        with tf.name_scope('optimizer'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            decay_steps = (self.num_epochs + 1) * self.num_iters_per_epoch
            lr = tf.train.polynomial_decay(self.learning_rate, global_step=global_step,
                                           decay_steps=decay_steps, end_learning_rate=0.00001, power=1.0)
            if self.update_rule == 'momentum':
                optimizer = self.optimizer(learning_rate=lr, momentum=0.9)
            else:
                optimizer = self.optimizer(learning_rate=lr)
            train_op = optimizer.minimize(loss, global_step=global_step)

        tf.get_variable_scope().reuse_variables()
        _, generated_sampled_caps = self.model.build_sampler(max_len=self.data_loader.seq_length - 1,
                                                             sampling_out=True)
        _, generated_greedy_caps = self.model.build_sampler(max_len=self.data_loader.seq_length - 1, max_out=True)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        tf.summary.scalar('learning_rate', lr)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()

        # GPU options
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        print("Single epoch process CIDEr optimization")
        print("Data size: %d" % self.num_train_img)
        print("Batch size: %d" % self.batch_size)
        print("Iterations per epoch: %d" % self.num_iters_per_epoch)

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            print("Start training previous last cross entropy loss Model..")
            print("Restore Model file : %s" % os.path.join(self.model_path, 'model-%d' % self.num_epochs))
            saver.restore(sess, os.path.join(self.model_path, 'model-%d' % self.num_epochs))

            curr_loss = 0
            start_t = time.time()

            # initianlize CIDEr evaluation
            self._init_cider_scorer()

            # shuffle training dataset
            self.data_loader.shuffle_data()

            # training
            for i in range(self.num_iters_per_epoch):
                data_batch = self.data_loader.get_batch('train', i)
                features_batch = data_batch['features']
                captions_batch = data_batch['labels']
                feed_dict = {self.model.features: features_batch,
                             self.model.captions: captions_batch}

                if (i + 1) % 100 == 0:
                    print_reward = True
                else:
                    print_reward = False

                # process CIDEr optimization
                greedy_cap = sess.run(generated_greedy_caps, feed_dict)
                sampled_cap = sess.run(generated_sampled_caps, feed_dict)
                reward = self._get_self_critical_reward(sampled_cap, greedy_cap, data_batch, print_reward)
                encoded_sampled_cap = normalized_encoding(sampled_cap, self.data_loader.word_to_idx)
                feed_dict = {self.model.features: features_batch,
                             self.model.captions: encoded_sampled_cap,
                             self.model.rewards: reward}
                _, l = sess.run([train_op, loss], feed_dict)
                curr_loss += l

                # write summary for tensorboard visualization
                if i % 10 == 0:
                    summary = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary, self.num_epochs * self.num_iters_per_epoch + i)

                if (i + 1) % self.print_every == 0:
                    print("\nTrain loss at RL & iteration %d (mini-batch): %.5f" % (i + 1, l))
                    ground_truths = data_batch['gts'][0]
                    decoded = decode_captions(ground_truths, self.model.idx_to_word)
                    for j, gt in enumerate(decoded):
                        print("Ground truth %d: %s" % (j + 1, gt))
                    gen_caps = sess.run(generated_greedy_caps, feed_dict)
                    decoded = decode_captions(gen_caps, self.model.idx_to_word)
                    print("Generated caption: %s\n" % decoded[0])

            print("Current epoch loss: ", curr_loss)
            print("Elapsed time: ", time.time() - start_t)

            # save model's parameters
            saver.save(sess, os.path.join(self.model_path, 'model_rl'), global_step=self.num_epochs)
            print("model_rl-%s saved." % self.num_epochs)

    def test(self, attention_visualization=True, rand_test_img=True, image_path='data/image', max_len=20, beam_size=3):
        # build a graph to sample captions
        # alphas, sampled_captions = self.model.build_sampler(max_len=max_len)
        alphas, sampled_captions, scores = self.model.build_sampler_beam(max_len=max_len, beam_size=beam_size)

        # GPU options
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            for i in range(self.data_loader.num_test_images):
                if rand_test_img:
                    data = self.data_loader.get_test_sample(-1)
                else:
                    data = self.data_loader.get_test_sample(i)
                features = data['features']
                feed_dict = {self.model.features: features}

                alps, caps, score = sess.run([alphas, sampled_captions, scores], feed_dict)
                ground_truth_cap = decode_captions(data['gts'][0], self.model.idx_to_word)
                decoded = decode_captions(caps[0], self.model.idx_to_word)

                print("Iter %d..." % (i+1))
                print("Img ID.", data['id'])
                for j, gt in enumerate(ground_truth_cap):
                    print("Ground truth %d: %s" % (j + 1, gt))
                for j in range(beam_size):
                    print("Generated Caption %d: %s (score: %.4f)" % (j + 1, decoded[j], score[0, j]))
                print()

                if attention_visualization:
                    img = skimage.io.imread(os.path.join(image_path, data['file_path']))
                    img = skimage.img_as_float(img)
                    if img.ndim == 2:
                        img = skimage.color.grey2rgb(img)
                    boxes = data['boxes'].astype(int)

                    # draw all box
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    plt.axis('off')
                    for b in range(boxes.shape[0]):
                        ax.add_patch(patches.Rectangle((boxes[b, 0], boxes[b, 1]),
                                                       boxes[b, 2]-boxes[b, 0], boxes[b, 3]-boxes[b, 1],
                                                       linewidth=1, edgecolor='r', facecolor='none'))
                    # plt.show()
                    # fig.clear()

                    # draw attention map
                    fig = plt.figure(figsize=(16, 8))
                    ax = fig.add_subplot(3, 6, 1)
                    ax.imshow(img)
                    plt.axis('off')
                    plt.title('Caption ID :{}'.format(data['id']), color='black',
                              backgroundcolor='white', fontsize=18)

                    # Plot images with attention weights
                    words = decoded[0].split(" ")
                    for t in range(len(words)):
                        if t > 16:
                            break

                        alphamap = np.zeros((img.shape[0], img.shape[1]))
                        for b in range(boxes.shape[0]):
                            alphamap[boxes[b, 1]:boxes[b, 3], boxes[b, 0]:boxes[b, 2]] += alps[0, 0, t, b]
                        max_idx = np.argmax(alps[0, 0, t, :])
                        att_img = np.dstack((img, alphamap))
                        ax = fig.add_subplot(3, 6, t + 2)
                        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
                        ax.imshow(att_img)
                        ax.add_patch(patches.Rectangle((boxes[max_idx, 0], boxes[max_idx, 1]),
                                                       boxes[max_idx, 2]-boxes[max_idx, 0],
                                                       boxes[max_idx, 3]-boxes[max_idx, 1],
                                                       linewidth=1, edgecolor='r', facecolor='none'))
                        plt.axis('off')
                    plt.show()

    def output_result_for_eval(self, max_len=20, beam_size=3):
        # build a graph to sample captions
        # _, sampled_captions = self.model.build_sampler(max_len=max_len)
        _, sampled_captions, score = self.model.build_sampler_beam(max_len=max_len, beam_size=beam_size)

        # GPU options
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            out = []
            for i in range(self.data_loader.num_test_images):
                data = self.data_loader.get_test_sample(i)
                features = data['features']
                feed_dict = {self.model.features: features}

                sam_cap, s = sess.run([sampled_captions, score], feed_dict)
                decoded = decode_captions(sam_cap[0][0], self.model.idx_to_word)

                for j, gen in enumerate(decoded):
                    cap_info = {}
                    cap_info['image_id'] = data['id']
                    cap_info['caption'] = decoded[j]
                    out.append(cap_info)

            json.dump(out, open('./result.json', 'w'))
