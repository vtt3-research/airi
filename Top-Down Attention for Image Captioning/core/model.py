from __future__ import division

import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, word_to_idx, num_features=36, dim_feature=2048, dim_embed=1000,
                 dim_hidden=1000, dim_attention=512, n_time_step=16):
        """
        Args:
            word_to_idx: Word-to-index mapping dictionary.
            num_features: Number of features(resions) in a image.
            dim_feature: Dimension of feature vectors.
            dim_embed: Dimension of word embedding.
            dim_hidden: Dimension of hidden state in each LSTM.
            dim_attention: Dimension of hidden state in attention layer.
            n_time_step: Time step size of LSTM.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.V = len(word_to_idx)  # Vocabulary size
        self.L = num_features  # Number of features(regions) per image
        self.D = dim_feature  # Dimension of features
        self.E = dim_embed  # Size of word embedding
        self.H = dim_hidden  # Number of hidden units in each LSTM
        self.A = dim_attention  # Number of hidden units in attention layer
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']

        # Initailizer
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
        self.rewards = tf.placeholder(tf.float32, [None])

    @staticmethod
    def _get_mean_pooled_feature(features):
        global_feature = tf.reduce_mean(features, axis=1)
        return global_feature

    def _get_initial_lstm(self, inputs):
        with tf.variable_scope('initial_lstm'):
            w_h1 = tf.get_variable('w_h1', [self.D, self.H], initializer=self.weight_initializer)
            b_h1 = tf.get_variable('b_h1', [self.H], initializer=self.const_initializer)
            w_c1 = tf.get_variable('w_c1', [self.D, self.H], initializer=self.weight_initializer)
            b_c1 = tf.get_variable('b_c1', [self.H], initializer=self.const_initializer)

            w_h2 = tf.get_variable('w_h2', [self.D, self.H], initializer=self.weight_initializer)
            b_h2 = tf.get_variable('b_h2', [self.H], initializer=self.const_initializer)
            w_c2 = tf.get_variable('w_c2', [self.D, self.H], initializer=self.weight_initializer)
            b_c2 = tf.get_variable('b_c2', [self.H], initializer=self.const_initializer)

            h1 = tf.nn.tanh(tf.matmul(inputs, w_h1) + b_h1)  # (N, H)
            c1 = tf.nn.tanh(tf.matmul(inputs, w_c1) + b_c1)  # (N, H)
            h2 = tf.nn.tanh(tf.matmul(inputs, w_h2) + b_h2)  # (N, H)
            c2 = tf.nn.tanh(tf.matmul(inputs, w_c2) + b_c2)  # (N, H)

            return c1, h1, c2, h2

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.E], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, E)
            return x

    # See paper's section 3.2.1 and figure 3.
    def _topdown_attention_lstm(self, h2, v, w, c, h1, lstm_cell, reuse=False):
        with tf.variable_scope('topdown_attention_lstm', reuse=reuse):
            w_x = tf.get_variable('w_x', [(self.H + self.D + self.E), self.H], initializer=self.weight_initializer)
            b_x = tf.get_variable('b_x', [self.H], initializer=self.const_initializer)

            # Concatenate inputs (Paper's Eq.2)
            x = tf.concat(values=[h2, v, w], axis=1)  # (N, (H + D + E))

            # Attention LSTM
            x = tf.matmul(x, w_x) + b_x
            _, (c, h1) = lstm_cell(inputs=x, state=[c, h1])

            return c, h1

    def _attention_layer(self, features, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w_v = tf.get_variable('w_v', [self.D, self.A], initializer=self.weight_initializer)
            b_v = tf.get_variable('b_v', [self.A], initializer=self.const_initializer)
            w_h = tf.get_variable('w_h', [self.H, self.A], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.A], initializer=self.const_initializer)
            w_a = tf.get_variable('w_a', [self.A, 1], initializer=self.weight_initializer)

            # Embedding image features (Paper's Eq.3)
            features_flat = tf.reshape(features, [-1, self.D])  # (N * L, D)
            embed_features = tf.matmul(features_flat, w_v) + b_v  # (N * L, A)
            embed_features = tf.reshape(embed_features, [-1, self.L, self.A])  # (N, L, A)

            # Embedding hidden state 1 (Paper's Eq.3)
            embed_h = tf.matmul(h, w_h) + b_h  # (N, A)
            embed_h = tf.expand_dims(embed_h, axis=1)  # (N, 1, A)

            # Add WV and Wh (Paper's Eq.3)
            embed_h_tile = tf.tile(embed_h, [1, self.L, 1])  # (N, L, A)
            add_embeddings = tf.add(embed_features, embed_h_tile)

            # Applied tanh and mul w_a (Paper's Eq.3)
            tanh_embeddings = tf.nn.tanh(add_embeddings)
            a = tf.reshape(tf.matmul(tf.reshape(tanh_embeddings, [-1, self.A]), w_a), [-1, self.L])  # (N, L)

            # Applied softmax (Paper's Eq.4)
            alpha = tf.nn.softmax(a, name='attention_weight')  # (N, L)

            # Weighted sum (Paper's Eq.5)
            alpha_tile = tf.tile(tf.expand_dims(alpha, 2), [1, 1, self.D])  # (N, L, D)
            v_hat = tf.reduce_sum(tf.multiply(features, alpha_tile), axis=1)  # (N, D)

            return v_hat, alpha

    # See paper's section 3.2.2 and figure 3.
    def _language_lstm(self, v, h1, c, h2, lstm_cell, reuse=False):
        with tf.variable_scope('langauge_lstm', reuse=reuse):
            w_x = tf.get_variable('w_x', [(self.D + self.H), self.H], initializer=self.weight_initializer)
            b_x = tf.get_variable('b_x', [self.H], initializer=self.const_initializer)

            # Concatenate inputs (Paper's Eq.6)
            x = tf.concat(values=[v, h1], axis=1)  # (N, (H + D + E))

            # Langauge LSTM
            x = tf.matmul(x, w_x) + b_x
            _, (c, h2) = lstm_cell(inputs=x, state=[c, h2])

            return c, h2

    def _prob_layer(self, h2, reuse=False):
        with tf.variable_scope('prob_layer', reuse=reuse):
            w_p = tf.get_variable('w_p', [self.H, self.V], initializer=self.weight_initializer)
            b_p = tf.get_variable('b_p', [self.V], initializer=self.const_initializer)

            # Applied softmax (Paper's Eq.7)
            logits = tf.matmul(h2, w_p) + b_p
            p = tf.nn.softmax(logits)

            return logits, p

    @staticmethod
    def _batch_norm(x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    # Separate cross-entropy model and reinforcement model.
    # Even if you train network with cross-entropy loss (not reinforce part),
    # you spent a lot of time (maybe 20 times?)
    # because reinforce model part drawn on a graph additionally.
    def build_model_xe(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        # mean-pooled image features
        mean_features = self._get_mean_pooled_feature(features)

        # initialize states
        c1, h1, c2, h2 = self._get_initial_lstm(mean_features)

        # word embedding
        w = self._word_embedding(captions_in)

        loss_batch = tf.zeros([batch_size], dtype=tf.float32)
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            c1, h1 = self._topdown_attention_lstm(h2, mean_features, w[:, t, :], c1, h1, lstm_cell1, reuse=(t != 0))
            v, alpha = self._attention_layer(features, h1, reuse=(t != 0))
            c2, h2 = self._language_lstm(v, h1, c2, h2, lstm_cell2, reuse=(t != 0))
            logits, _ = self._prob_layer(h2, reuse=(t != 0))

            loss_batch += tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=captions_out[:, t]) * mask[:, t]

        loss = tf.reduce_sum(loss_batch)

        return loss / tf.to_float(batch_size)

    def build_model_reinforce(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        # mean-pooled image features
        mean_features = self._get_mean_pooled_feature(features)

        # initialize states
        c1, h1, c2, h2 = self._get_initial_lstm(mean_features)

        # word embedding
        w = self._word_embedding(captions_in)

        loss_batch = tf.zeros([batch_size], dtype=tf.float32)
        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            c1, h1 = self._topdown_attention_lstm(h2, mean_features, w[:, t, :], c1, h1, lstm_cell1, reuse=(t != 0))
            v, alpha = self._attention_layer(features, h1, reuse=(t != 0))
            c2, h2 = self._language_lstm(v, h1, c2, h2, lstm_cell2, reuse=(t != 0))
            logits, p = self._prob_layer(h2, reuse=(t != 0))

            # for REINFORCE learning
            idx_rl = captions_out[:, t]
            idx_rl_flattened = tf.range(0, batch_size) * self.V + idx_rl
            log_softmax_flattened = tf.reshape(tf.log(p), [-1])
            partitions = tf.reduce_sum(tf.one_hot(idx_rl_flattened, tf.shape(log_softmax_flattened)[0],
                                                  dtype=tf.int32), 0)
            loss_batch += tf.dynamic_partition(log_softmax_flattened, partitions, 2)[1] * mask[:, t]

        loss = tf.reduce_sum(-1 * self.rewards * loss_batch)

        return loss / tf.to_float(batch_size)

    # Choice decoding mode
    # If max_out variable is True, then captions generate by greedily decoding
    #   (output word with highest probability at a time).
    # If sampling_out variable is True, then captions generate by sampling
    #   from the entire probability distribution.
    # If both are False, then captions generate by greedily decoding
    #   but enforce that single word cannot be predicted twice in a row.
    def build_sampler(self, max_len=20, max_out=False, sampling_out=False):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        # mean-pooled image features
        mean_features = self._get_mean_pooled_feature(features)

        # initialize states
        c1, h1, c2, h2 = self._get_initial_lstm(mean_features)

        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        sampled_word_list = []
        alpha_list = []
        sampled_word = None

        for t in range(max_len):
            if t == 0:
                w = self._word_embedding(tf.fill([tf.shape(features)[0]], self._start))
            else:
                w = self._word_embedding(inputs=sampled_word, reuse=True)

            c1, h1 = self._topdown_attention_lstm(h2, mean_features, w, c1, h1, lstm_cell1, reuse=(t != 0))
            v, alpha = self._attention_layer(features, h1, reuse=(t != 0))
            c2, h2 = self._language_lstm(v, h1, c2, h2, lstm_cell2, reuse=(t != 0))
            _, p = self._prob_layer(h2, reuse=(t != 0))

            if max_out:  # greedily decoding
                sampled_word = tf.argmax(p, 1)
            elif sampling_out:  # probability sampling
                samples = tf.multinomial(tf.log(p), 1)
                sampled_word = tf.cast(samples[:, 0], tf.int32)
            else:  # enforce that single word cannot be predicted twice in a row
                if t == 0:
                    sampled_word = tf.argmax(p, 1)
                else:
                    arg_max_word = tf.argmax(p, 1)
                    _, cand_word = tf.nn.top_k(p, k=2, sorted=True)
                    cand_word = tf.cast(cand_word, dtype=tf.int64)
                    sampled_word = tf.where(tf.equal(sampled_word, arg_max_word), cand_word[:, 1], arg_max_word)

            alpha_list.append(alpha)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, sampled_captions

    def _stretch_to_beam_size(self, beam_size, word_idx, alpha, c1, c2, h1, h2, features, mean_feature):
        # store word and tile state, features (times beam_size)
        sampled_word_list = tf.expand_dims(tf.expand_dims(word_idx[:, 0], 1), 2)
        alpha_list = tf.expand_dims(tf.expand_dims(alpha, 1), 2)
        state_c1_list = tf.expand_dims(c1, 1)
        state_c2_list = tf.expand_dims(c2, 1)
        state_h1_list = tf.expand_dims(h1, 1)
        state_h2_list = tf.expand_dims(h2, 1)
        features_list = tf.expand_dims(features, 1)
        mean_feature_list = tf.expand_dims(mean_feature, 1)
        for b in range(1, beam_size):
            sampled_word_list = tf.concat(axis=1,
                                          values=[sampled_word_list,
                                                  tf.expand_dims(tf.expand_dims(word_idx[:, b], 1), 2)])
            alpha_list = tf.concat(axis=1,
                                   values=[alpha_list, tf.expand_dims(tf.expand_dims(alpha, 1), 2)])
            state_c1_list = tf.concat(axis=1,
                                      values=[state_c1_list, tf.expand_dims(c1, 1)])
            state_c2_list = tf.concat(axis=1,
                                      values=[state_c2_list, tf.expand_dims(c2, 1)])
            state_h1_list = tf.concat(axis=1,
                                      values=[state_h1_list, tf.expand_dims(h1, 1)])
            state_h2_list = tf.concat(axis=1,
                                      values=[state_h2_list, tf.expand_dims(h2, 1)])
            features_list = tf.concat(axis=1, values=[features_list, tf.expand_dims(features, 1)])
            mean_feature_list = tf.concat(axis=1, values=[mean_feature_list,
                                                          tf.expand_dims(mean_feature, 1)])

        features = tf.reshape(features_list, [-1, self.L, self.D])
        mean_feature = tf.reshape(mean_feature_list, [-1, self.D])
        c1 = tf.reshape(state_c1_list, [-1, self.H])
        c2 = tf.reshape(state_c2_list, [-1, self.H])
        h1 = tf.reshape(state_h1_list, [-1, self.H])
        h2 = tf.reshape(state_h2_list, [-1, self.H])

        # end score matrix to no seek to beam search the finished sentence
        end_score_fore = tf.fill([tf.shape(features)[0], self._end], tf.constant(-999.0))
        end_score_back = tf.fill([tf.shape(features)[0], self.V - self._end - 1], tf.constant(-999.0))

        return sampled_word_list, alpha_list, features, mean_feature, c1, c2, h1, h2, end_score_fore, end_score_back

    def _update_beam_search(self, beam_size, feature_idx, word_idx, alpha, sampled_word_list, alpha_list):
        # update beam search
        new_sampled_word_list = None
        new_alpha_list = None
        for b in range(beam_size):
            prev_idx = word_idx[:, b] // self.V
            prev_word = tf.expand_dims(tf.gather_nd(sampled_word_list, tf.stack((feature_idx, prev_idx), -1)),
                                       1)
            prev_alpha = tf.expand_dims(tf.gather_nd(alpha_list, tf.stack((feature_idx, prev_idx), -1)), 1)
            merge_word = tf.concat(axis=2, values=[prev_word,
                                                   tf.expand_dims(tf.expand_dims(word_idx[:, b] % self.V, 1),
                                                                  2)])
            merge_alpha = tf.concat(axis=2, values=[prev_alpha, tf.expand_dims(
                tf.expand_dims(tf.gather(alpha, prev_idx), 1), 2)])
            if b == 0:
                new_sampled_word_list = merge_word
                new_alpha_list = merge_alpha
            else:
                new_sampled_word_list = tf.concat(axis=1, values=[new_sampled_word_list, merge_word])
                new_alpha_list = tf.concat(axis=1, values=[new_alpha_list, merge_alpha])

        sampled_word_list = new_sampled_word_list
        alpha_list = new_alpha_list

        return sampled_word_list, alpha_list

    def build_sampler_beam(self, max_len=20, beam_size=5):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        # mean-pooled image features
        mean_feature = self._get_mean_pooled_feature(features)

        # initialize states
        c1, h1, c2, h2 = self._get_initial_lstm(mean_feature)

        lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        # batch index
        feature_idx = tf.range(tf.shape(features)[0])

        sampled_word_list = []
        alpha_list = []
        word_score = 0.0
        end_score_fore = None
        end_score_back = None

        for t in range(max_len):
            if t == 0:
                w = self._word_embedding(tf.fill([tf.shape(features)[0]], self._start))
                c1, h1 = self._topdown_attention_lstm(h2, mean_feature, w, c1, h1, lstm_cell1, reuse=False)
                v, alpha = self._attention_layer(features, h1, reuse=False)
                c2, h2 = self._language_lstm(v, h1, c2, h2, lstm_cell2, reuse=False)
                _, p = self._prob_layer(h2, reuse=False)

                score = tf.log(p)
                word_score, word_idx = tf.nn.top_k(score, k=beam_size, sorted=True)
                sampled_word_list, alpha_list, features, mean_feature, c1, c2, h1, h2, end_score_fore, end_score_back \
                    = self._stretch_to_beam_size(beam_size, word_idx, alpha, c1, c2, h1, h2, features, mean_feature)
            else:
                # concatenate end score matrixs and tiling previous word
                sampled_word = tf.reshape(sampled_word_list[:, :, -1], [-1])
                prev_word_tile = tf.tile(tf.expand_dims(sampled_word, 1), [1, self.V])
                prev_score = tf.expand_dims(tf.reshape(word_score, [-1]), 1)
                prev_score_tile = tf.tile(tf.expand_dims(tf.reshape(word_score, [-1]), 1), [1, self.V])
                end_score_mat = tf.concat(axis=1, values=[end_score_fore, prev_score, end_score_back])

                w = self._word_embedding(inputs=sampled_word, reuse=True)
                c1, h1 = self._topdown_attention_lstm(h2, mean_feature, w, c1, h1, lstm_cell1, reuse=True)
                v, alpha = self._attention_layer(features, h1, reuse=True)
                c2, h2 = self._language_lstm(v, h1, c2, h2, lstm_cell2, reuse=True)
                _, p = self._prob_layer(h2, reuse=True)

                score = prev_score_tile + tf.log(p)

                # update score with no finished sentence
                score_non_finished = tf.where(tf.equal(prev_word_tile, self._end), end_score_mat, score)
                word_score, word_idx = tf.nn.top_k(tf.reshape(score_non_finished, [-1, beam_size * self.V]),
                                                   k=beam_size, sorted=True)
                sampled_word_list, alpha_list \
                    = self._update_beam_search(beam_size, feature_idx, word_idx, alpha, sampled_word_list, alpha_list)

        return alpha_list, sampled_word_list, tf.exp(word_score)
