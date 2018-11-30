from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class C3D(nn.Module):

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input shape : (batch size, ch=3, num frames=16, h=112, w=112)
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        # output shape : (batch size, 64, 16, 56, 56)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        # output shape : (batch size, 128, 8, 28, 28)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        # output shape : (batch size, 256, 4, 14, 14)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        # output shape : (batch size, 512, 2, 7, 7)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        # output shape : (batch size, 512, 1, 4, 4)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.fc7(h)

        return h


class AttDetector(nn.Module):
    def __init__(self, in_c, num_class):
        super(AttDetector, self).__init__()

        self.fc1 = nn.Linear(in_c, 1024)
        self.fc2 = nn.Linear(1024, num_class)

        for m in self.modules():
            if type(m) in [nn.Linear]:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        if self.training:
            return x
        else:
            return F.softmax(x, dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class TEP(nn.Module):

    def __init__(self, in_c, num_anchor_layer, num_scale_ratio, num_class):
        super(TEP, self).__init__()

        # base layers
        self.base_layer = nn.Sequential(
            ConvBlock(in_c, 2048, 3, p=1),
            ConvBlock(2048, 1024, 3, p=1),
        )

        # anchor layers
        self.num_anchor = num_anchor_layer
        anchor_layers = []
        in_c = 1024
        out_c = 512
        for i in range(self.num_anchor):
            anchor_layers += [nn.Conv1d(in_c, out_c, kernel_size=1)]
            anchor_layers += [nn.Conv1d(out_c, out_c, kernel_size=3, stride=2, padding=1)]
            in_c = out_c
        self.anchor_dim = out_c
        self.anchor_layers = nn.ModuleList(anchor_layers)

        # prediction layers
        self.num_scale_ratio = num_scale_ratio
        self.num_class = num_class
        loc_layers = []  # box location
        evn_layers = []  # event/background (event: positive proposal)
        cls_layers = []  # classification
        for i in range(self.num_anchor):
            loc_layers += [nn.Conv1d(512, 2*self.num_scale_ratio, kernel_size=1)]
            evn_layers += [nn.Conv1d(512, 2*self.num_scale_ratio, kernel_size=1)]
            cls_layers += [nn.Conv1d(512, self.num_class*self.num_scale_ratio, kernel_size=1)]
        self.loc_layers = nn.ModuleList(loc_layers)
        self.evn_layers = nn.ModuleList(evn_layers)
        self.cls_layers = nn.ModuleList(cls_layers)

        # weight initialize
        for m in self.modules():
            if type(m) in [nn.Conv1d, nn.Linear]:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.size(0) != 1:
            raise ValueError("Batch size must be 1")
        source = list()
        loc = list()
        evn = list()
        cls = list()
        temporal_dims = list()

        # base layers
        # input shape : (batch size, ch=4096, N)
        x = self.base_layer(x)
        # output shape : (batch size, 1024, N)

        # anchor layers
        for i, v in enumerate(self.anchor_layers):
            if x.size(2) < 3:
                break
            x = F.relu(v(x), inplace=True)
            if i % 2 == 1:
                source.append(x)
                temporal_dims.append(x.size(2))
        # output shape : (batch size, 512, N/(2^n))

        # prediction layers
        for (s, l, p, c) in zip(source, self.loc_layers, self.evn_layers, self.cls_layers):
            loc.append(l(s).permute(0, 2, 1).contiguous())
            evn.append(p(s).permute(0, 2, 1).contiguous())
            cls.append(c(s).permute(0, 2, 1).contiguous())

        # from list to torch tensor
        loc = torch.cat([o for o in loc], 1)
        evn = torch.cat([o for o in evn], 1)
        cls = torch.cat([o for o in cls], 1)
        source = torch.cat([o for o in source], 2)
        source = source.view(-1, 1).repeat(1, self.num_scale_ratio).view(x.shape[0], 512, -1)

        if self.training:
            output = (
                loc.view(loc.size(0), -1, 2),
                evn.view(evn.size(0), -1, 2),
                cls.view(cls.size(0), -1, self.num_class),
                source,
                temporal_dims
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 2),
                F.softmax(evn.view(evn.size(0), -1, 2), dim=2),
                F.softmax(cls.view(cls.size(0), -1, self.num_class), dim=2),
                source,
                temporal_dims
            )

        return output


class SG(nn.Module):

    def __init__(self, in_feats_dim, voca_size,
                 caps_length, attribute_size,
                 embedding_dim, hidden_dim, use_gpu):
        super(SG, self).__init__()

        self.voca_size = voca_size
        self.lstm_length = caps_length
        self.hidden_dim = hidden_dim
        self.attention_dim = embedding_dim
        self.use_gpu = use_gpu
        first_lstm_dim = hidden_dim+in_feats_dim+embedding_dim
        second_lstm_dim = hidden_dim+in_feats_dim

        # sentence generation
        self.attribute_emb = nn.Linear(attribute_size, second_lstm_dim)
        self.word_emb = nn.Embedding(voca_size, embedding_dim)
        self.lstm1 = nn.LSTMCell(first_lstm_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(second_lstm_dim, hidden_dim)
        self.prob_layer = nn.Linear(hidden_dim, voca_size)

        # attention layer
        self.attention_linear_f = nn.Linear(in_feats_dim, self.attention_dim)
        self.attention_linear_h = nn.Linear(hidden_dim, self.attention_dim)
        self.attention_linear_a = nn.Linear(self.attention_dim, 1)
        self.attention_tanh = nn.Tanh()
        self.attention_softmax = nn.Softmax(dim=1)

        # weight initialize
        for m in self.modules():
            if type(m) in [nn.LSTM, nn.Linear]:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_feats, input_atts, input_sents=None):
        # 3++ dimension PyTorch tensor construction very slow
        # (Need to (batch_size, sentence_length, voca_size) tensor
        # So, construct empty list and append for each iteration
        # And last, call stack function for type casting from list to tensor
        outputs = []
        output_preds = []

        # hidden/cell state initialize
        h1_t, c1_t = self._init_hidden(input_feats.size(0))
        h2_t, c2_t = self._init_hidden(input_feats.size(0))
        # if self.use_gpu:
        #     h1_t = h1_t.cuda()
        #     c1_t = c1_t.cuda()
        #     h2_t = h2_t.cuda()
        #     c2_t = c2_t.cuda()

        # input shape
        # input feats : (batch size=N, feature_dim=F, num_features=L)

        # mean feature
        mean_feature = torch.mean(input_feats, 2)  # -> (N, F)

        # 1st step (using attribute)
        embed_attribute = self.attribute_emb(input_atts)
        h2_t, c2_t = self.lstm2(embed_attribute, (h2_t, c2_t))

        # 2rd ~ last step
        for i in range(self.lstm_length-1):
            if self.training:
                it = input_sents[:, i]
            elif i == 0:
                it = torch.ones(input_feats.size(0)).long()  # <START> token index is 1
                if self.use_gpu:
                    it = it.cuda()
            else:
                it = gen_word

            # word embedding
            word_vec = self.word_emb(it)

            # 1st LSTM layer
            h1_t, c1_t = self.lstm1(torch.cat([h2_t, mean_feature, word_vec], dim=1), (h1_t, c1_t))

            # attention layer
            att_vec = self._attention_layer(input_feats, h1_t)  # (N, F)

            # 2nd LSTM layer
            h2_t, c2_t = self.lstm2(torch.cat([h1_t, att_vec], dim=1), (h2_t, c2_t))

            # prediction layer
            predict = self.prob_layer(h2_t)
            _, gen_word = torch.max(F.log_softmax(predict, dim=1).data, 1)  # Greedy sampling
            gen_word = gen_word.view(-1).long()

            # Check finished sentence generation for all batch
            if i >= 1 and torch.sum(it) == 0:
                for j in range(i, self.lstm_length-1):
                    if self.use_gpu:
                        outputs += [torch.zeros(input_feats.size(0)).long().cuda()]
                        output_preds += [torch.zeros(input_feats.size(0), self.voca_size).float().cuda()]
                    else:
                        outputs += [torch.zeros(input_feats.size(0)).long()]
                        output_preds += [torch.zeros(input_feats.size(0), self.voca_size).float()]
                break
            outputs += [gen_word]
            output_preds += [predict]

        # Compact the list of predictions
        outputs = torch.stack(outputs, 1).squeeze(1)
        output_preds = torch.stack(output_preds, 1).squeeze(2)
        return outputs, output_preds

    def sample(self, input_feats, input_atts, greedy=True):
        outputs = []
        output_logprobs = []

        h1_t, c1_t = self._init_hidden(input_feats.size(0))
        h2_t, c2_t = self._init_hidden(input_feats.size(0))
        # if self.use_gpu:
        #     h1_t = h1_t.cuda()
        #     c1_t = c1_t.cuda()
        #     h2_t = h2_t.cuda()
        #     c2_t = c2_t.cuda()

        mean_feature = torch.mean(input_feats, 2)

        embed_attribute = self.attribute_emb(input_atts)
        h2_t, c2_t = self.lstm2(embed_attribute, (h2_t, c2_t))

        for i in range(self.lstm_length):
            if i == 0:  # Generate start token
                it = torch.ones(input_feats.size(0)).long()
                if self.use_gpu:
                    it = it.cuda()
            elif greedy:  # Greedy sampling
                sample_logprobs, it = torch.max(logprobs.data, 1)
                if self.use_gpu:
                    it = it.cuda()
                it = it.view(-1).long()
            else:  # MC sampling
                prob_prev = torch.exp(logprobs.data)
                it = torch.multinomial(prob_prev, 1)
                if self.use_gpu:
                    it = it.cuda()
                sample_logprobs = logprobs.gather(1, it).view(-1)
                it = it.view(-1).long()

            if i == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)

            word_vec = self.word_emb(it)
            h1_t, c1_t = self.lstm1(torch.cat([h2_t, mean_feature, word_vec], dim=1), (h1_t, c1_t))
            att_vec = self._attention_layer(input_feats, h1_t)
            h2_t, c2_t = self.lstm2(torch.cat([h1_t, att_vec], dim=1), (h2_t, c2_t))
            predict = self.prob_layer(h2_t)
            logprobs = F.log_softmax(predict, dim=1)

            # Check finished sentence generation for all batch
            if torch.sum(unfinished) == 0:
                for j in range(i, self.lstm_length):
                    if self.use_gpu:
                        outputs.append(torch.zeros(input_feats.size(0)).long().cuda())
                        output_logprobs.append(torch.zeros(input_feats.size(0)).float().cuda())
                    else:
                        outputs.append(torch.zeros(input_feats.size(0)).long())
                        output_logprobs.append(torch.zeros(input_feats.size(0)).float())
                break
            if i > 0:
                outputs.append(it)
                output_logprobs.append(sample_logprobs)

        outputs = torch.stack(outputs, 1)
        output_logprobs = torch.stack(output_logprobs, 1)
        return outputs, output_logprobs

    def _init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(batch_size, self.hidden_dim), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(batch_size, self.hidden_dim), requires_grad=False)
        return hidden.zero_(), cell.zero_()

    def _attention_layer(self, features, h):
        # shape
        # features : (batch size=N, feature_dim=F, num_feature=L)
        # h : (batch size=N, hidden_dim=H)

        # embedding image features
        embed_f = features.permute(0, 2, 1).contiguous()  # (N, L, F)
        embed_f = embed_f.view(-1, features.size(1))  # (N*L, F)
        embed_f = self.attention_linear_f(embed_f)  # (N*L, A)
        embed_f = embed_f.view(-1, features.size(2), self.attention_dim)  # (N, L, A)

        # embedding hidden state
        embed_h = self.attention_linear_h(h)  # (N, A)
        embed_h = embed_h.unsqueeze(1)  # (N, 1, A)

        # add embed features and embed hidden state
        embed_h = embed_h.repeat(1, features.size(2), 1)  # (N, L, A)
        a = embed_f + embed_h

        # applied tanh
        a = self.attention_tanh(a)

        # mul weight
        a = a.view(-1, self.attention_dim)  # (N*L, A)
        a = self.attention_linear_a(a)  # (N*L, 1)
        a = a.view(-1, features.size(2))  # (N, L)

        # softmax
        a = self.attention_softmax(a)

        # weighted attention sum
        a = a.unsqueeze(1)  # (N, 1, L)
        a = a.repeat(1, features.size(1), 1)  # (N, F, L)
        v_hat = a*features
        v_hat = v_hat.sum(2)  # (N, F)

        return v_hat


def build_att_model(num_class):
    return AttDetector(in_c=4096, num_class=num_class)


def build_tep_model(num_class):
    scale_ratios = np.asarray([0.7, 1.0, 1.3])

    return TEP(in_c=4096, num_anchor_layer=9,
               num_scale_ratio=scale_ratios.shape[0],
               num_class=num_class), \
        scale_ratios


def build_att_sg_model(num_class, voca_size, caps_length,
                       embedding_dim, hidden_dim, use_gpu):
    return AttDetector(in_c=4096, num_class=num_class), \
           SG(in_feats_dim=4096, voca_size=voca_size,
              caps_length=caps_length, attribute_size=num_class,
              embedding_dim=embedding_dim,
              hidden_dim=hidden_dim, use_gpu=use_gpu)


def build_model(num_class, voca_size, caps_length,
                embedding_dim, hidden_dim, use_gpu):
    scale_ratios = np.asarray([0.7, 1.0, 1.3])

    return C3D(), \
           AttDetector(in_c=4096, num_class=num_class), \
           TEP(in_c=4096, num_anchor_layer=9,
               num_scale_ratio=scale_ratios.shape[0],
               num_class=num_class), \
           SG(in_feats_dim=4096, voca_size=voca_size,
              caps_length=caps_length, attribute_size=num_class,
              embedding_dim=embedding_dim,
              hidden_dim=hidden_dim, use_gpu=use_gpu), \
           scale_ratios