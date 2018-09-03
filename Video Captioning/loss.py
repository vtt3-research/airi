from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DVCLoss(nn.Module):

    def __init__(self, scale_ratios, in_c=500, threshold=0.7,
                 alpha=0.5, beta=10.0, alpha1=0.1, alpha2=0.1,
                 lambda1=1.0, lambda2=20.0, use_gpu=True):
        super(DVCLoss, self).__init__()
        self.scale_ratios = scale_ratios
        self.in_c = in_c
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_gpu = use_gpu

    def forward(self, predictions, t_box, matches, events, sent_reward,
                sample_logprobs=None, gen_result=None, self_reward=None,
                gen_preds=None, gt_sent=None, rl_flag=True):
        loc, des, cls, temporal_dim = predictions

        loc_t = torch.Tensor(t_box.shape[0], 2)  # targets center, width
        event_t = torch.Tensor(t_box.shape[0]).long()  # distinguish event or background
        loc_p = torch.Tensor(t_box.shape[0], 2)  # predictions center, width

        predictions_center = t_box[:, 0] + self.alpha1 * t_box[:, 1] * loc[0].data[:, 0]
        predictions_width = t_box[:, 1] * torch.exp(self.alpha2 * loc[0].data[:, 1])

        loc_p[:, 0] = predictions_center
        loc_p[:, 1] = predictions_width
        loc_t[:, :] = torch.from_numpy(matches[:, :2]).float()
        event_t[:] = torch.from_numpy(events).long()

        if self.use_gpu:
            loc_t = loc_t.cuda()
            event_t = event_t.cuda()
            loc_p = loc_p.cuda()
        loc_t = Variable(loc_t, requires_grad=False)
        event_t = Variable(event_t, requires_grad=False)
        loc_p = Variable(loc_p)

        # positive proposals
        pos_idx = event_t > 0
        num_pos = torch.sum(event_t)
        loc_p_pos = loc_p[pos_idx]
        loc_t_pos = loc_t[pos_idx]

        # event/background classification loss
        cls = cls.view(-1, 2)
        loss_event = F.cross_entropy(cls, event_t)

        # temporal coordinate regression loss
        loss_tcr = F.smooth_l1_loss(loc_p_pos[:, 0], loc_t_pos[:, 0]) \
            + F.smooth_l1_loss(loc_p_pos[:, 1], loc_t_pos[:, 1])

        # descriptiveness regression loss
        des = des.view(-1)
        diff = des - sent_reward
        loss_des = torch.sum(diff * diff) / float(t_box.shape[0])

        # self-critical reward loss or cross entropy loss for sentence generator
        if rl_flag:
            if sample_logprobs is None or gen_result is None or self_reward is None:
                raise Exception()
            sample_logprobs = sample_logprobs.contiguous().view(-1)
            self_reward = self_reward.contiguous().view(-1)
            mask = (gen_result > 0).float()
            mask = torch.cat([mask.new(mask.shape[0], 1).fill_(1), mask[:, :-1]], 1).view(-1)
            loss_self_reward = - sample_logprobs * self_reward * mask
            loss_self_reward = torch.sum(loss_self_reward) / torch.sum(mask)
        else:
            if gen_preds is None or gt_sent is None:
                raise Exception()
            loss_self_reward = F.cross_entropy(gen_preds, gt_sent)

        loss = self.lambda1 * (loss_event + self.alpha * loss_tcr + self.beta * loss_des) \
            + self.lambda2 * loss_self_reward
        return loss, loss_event, loss_tcr, loss_des, loss_self_reward
