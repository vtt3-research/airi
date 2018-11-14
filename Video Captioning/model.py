from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


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


class DVCTEP(nn.Module):

    def __init__(self, in_c, num_anchor_layer, num_scale_ratio):
        super(DVCTEP, self).__init__()

        # base layers
        self.base_layer = nn.Sequential(
            ConvBlock(in_c, 2048, 3, p=1),
            ConvBlock(2048, 1024, 3, s=2, p=1),
        )

        # anchor layers
        self.num_anchor = num_anchor_layer
        anchor_layers = []
        in_c = 1024
        out_c = 512
        for i in range(self.num_anchor):
            anchor_layers += [ConvBlock(in_c, out_c, 3, s=2, p=1)]
            in_c = out_c
        self.anchor_layers = nn.ModuleList(anchor_layers)

        # prediction layers
        self.num_scale_ratio = num_scale_ratio
        loc_layers = []
        des_layers = []
        cls_layers = []
        for i in range(self.num_anchor):
            loc_layers += [nn.Linear(512, 2*self.num_scale_ratio)]
            des_layers += [nn.Linear(512, 1*self.num_scale_ratio)]
            cls_layers += [nn.Linear(512, 2*self.num_scale_ratio)]
        self.loc_layers = nn.ModuleList(loc_layers)
        self.des_layers = nn.ModuleList(des_layers)
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
        source = list()
        loc = list()
        des = list()
        cls = list()
        temporal_dims = list()

        # base layers
        x = self.base_layer(x)

        # anchor layers
        for i in range(self.num_anchor):
            if x.size(2) < 3:
                break
            x = self.anchor_layers[i](x)
            source.append(x)
            temporal_dims.append(x.size(2))

        # prediction layers
        for (s, l, d, c) in zip(source, self.loc_layers, self.des_layers, self.cls_layers):
            s = s.permute(0, 2, 1)
            s = s.view(s.size(0)*s.size(1), -1)
            loc.append(l(s).view(x.size(0), s.size(0), -1).contiguous())
            des.append(F.sigmoid(d(s)).view(x.size(0), s.size(0), -1).contiguous())
            cls.append(c(s).view(x.size(0), s.size(0), -1).contiguous())

        # from list to torch tensor
        loc = torch.cat([o for o in loc], 1)
        des = torch.cat([o for o in des], 1)
        cls = torch.cat([o for o in cls], 1)

        if self.training:
            output = (
                loc.view(loc.size(0), -1, 2),
                des.view(des.size(0), -1, 1),
                cls.view(cls.size(0), -1, 2),
                temporal_dims
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 2),
                des.view(des.size(0), -1, 1),
                F.softmax(cls.view(cls.size(0), -1, 2), dim=2),
                temporal_dims
            )

        return output


class DVCSG(nn.Module):

    def __init__(self, in_c, voca_size,
                 caps_length, attribute_size,
                 embedding_dim, hidden_dim, use_gpu):
        super(DVCSG, self).__init__()

        self.voca_size = voca_size
        self.lstm_length = caps_length
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # sentence generation
        self.attribute_emb = nn.Linear(attribute_size, embedding_dim)
        self.feature_emb = nn.Linear(in_c, embedding_dim)
        self.word_emb = nn.Embedding(voca_size, embedding_dim)
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.prob_layer = nn.Linear(hidden_dim, voca_size)

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
        h_t = Variable(torch.zeros(input_feats.size(0), self.embedding_dim).float())
        c_t = Variable(torch.zeros(input_feats.size(0), self.embedding_dim).float())
        if self.use_gpu:
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        # 1st LSTM
        att = self.attribute_emb(input_atts)
        h_t, c_t = self.lstm(att, (h_t, c_t))

        # 2nd LSTM
        feat = self.feature_emb(input_feats)
        h_t, c_t = self.lstm(feat, (h_t, c_t))

        # 3rd ~ last LSTM
        for i in range(self.lstm_length-1):
            if self.training:
                it = input_sents[:, i]
            elif i == 0:
                it = torch.ones(input_feats.size(0)).long()  # <START> token index is 1
                if self.use_gpu:
                    it = it.cuda()
            else:
                it = gen_word
            word_vec = self.word_emb(it)
            h_t, c_t = self.lstm(word_vec, (h_t, c_t))
            predict = self.prob_layer(h_t)
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

        h_t = Variable(torch.zeros(input_feats.size(0), self.embedding_dim).float())
        c_t = Variable(torch.zeros(input_feats.size(0), self.embedding_dim).float())
        if self.use_gpu:
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        att = self.attribute_emb(input_atts)
        h_t, c_t = self.lstm(att, (h_t, c_t))

        feat = self.feature_emb(input_feats)
        h_t, c_t = self.lstm(feat, (h_t, c_t))

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
            h_t, c_t = self.lstm(word_vec, (h_t, c_t))
            predict = self.prob_layer(h_t)
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


# Build attribute detector model
def build_att_model(in_c=500, num_class=200):
    return AttDetector(in_c, num_class)


# Build sentence generator model with attribute detector
def build_sg_model(in_c=500, num_class=200, voca_size=11122,
                   caps_length=32, embedding_dim=1024,
                   hidden_dim=1024, use_gpu=False):
    return AttDetector(in_c, num_class), \
       DVCSG(in_c, voca_size, caps_length, num_class,
             embedding_dim, hidden_dim, use_gpu)


# Build total model contained temporal event proposal, sentence generator and attribute detector
def build_models(in_c=500, num_class=200, voca_size=11122,
                 caps_length=32, embedding_dim=1024,
                 hidden_dim=1024, use_gpu=False):
    num_anchor_layer = 9
    scale_ratios = np.asarray([1, 1.25, 1.5])
    num_scale_ratio = scale_ratios.shape[0]
    return AttDetector(in_c, num_class), \
        DVCTEP(in_c, num_anchor_layer, num_scale_ratio), \
        DVCSG(in_c, voca_size, caps_length, num_class,
              embedding_dim, hidden_dim, use_gpu), \
        scale_ratios
