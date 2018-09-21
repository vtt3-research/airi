from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable

sys.path.insert(0, './coco-caption')
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu


Meteor_scorer = None
Cider_scorer = None
Bleu_scorer = None


def init_eval_metric():
    global Meteor_scorer
    global Cider_scorer
    global Bleu_scorer
    Meteor_scorer = Meteor_scorer or Meteor()
    Cider_scorer = Cider_scorer or Cider()
    Bleu_scorer = Bleu_scorer or Bleu(4)


def arr_to_str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] == 1:  # start token
            pass
        elif arr[i] == 2 or arr[i] == 0:  # end or null token
            break
        out += str(arr[i]) + ' '
    return out.strip()


def get_sc_reward(model, feats, sents, attribute, gen_result,
                  meteor_weight=1.0, cider_weight=0.0, bleu_weight=0.0):
    model.eval()
    with torch.no_grad():
        greedy_result, _ = model.sample(Variable(feats), Variable(attribute), greedy=True)
    model.train()

    gen_result = gen_result.data.cpu().numpy()
    greedy_result = greedy_result.data.cpu().numpy()
    sents = sents.data.cpu().numpy()

    res = OrderedDict()
    num_sample = gen_result.shape[0]
    for i in range(num_sample):
        res[i] = [arr_to_str(gen_result[i])]
    for i in range(num_sample):
        res[num_sample + i] = [arr_to_str(greedy_result[i])]

    gts = OrderedDict()
    for i in range(num_sample):
        gts[i] = [arr_to_str(sents[i])]

    res = {i: res[i] for i in range(2 * num_sample)}
    gts = {i: gts[i % num_sample] for i in range(2 * num_sample)}

    _, meteor_scores = Meteor_scorer.compute_score(gts, res)
    meteor_scores = np.asarray(meteor_scores)
    _, cider_scores = Cider_scorer.compute_score(gts, res)
    _, bleu_scores = Bleu_scorer.compute_score(gts, res)
    bleu_scores = np.array(bleu_scores[3])

    sent_lvl_reward = meteor_scores[:num_sample]

    scores = meteor_weight * meteor_scores \
             + cider_weight * cider_scores \
             + bleu_weight * bleu_scores
    self_scores = scores[:num_sample] - scores[num_sample:]
    self_rewards = np.repeat(self_scores[:, np.newaxis], gen_result.shape[1], 1)

    return sent_lvl_reward, self_rewards


def compute_meteor_score(gen_result, target):
    gen_result = gen_result.data.cpu().numpy()
    target = target.data.cpu().numpy()

    res = OrderedDict()
    num_sample = gen_result.shape[0]
    for i in range(num_sample):
        res[i] = [arr_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(num_sample):
        gts[i] = [arr_to_str(target[i])]

    res = {i: res[i] for i in range(num_sample)}
    gts = {i: gts[i] for i in range(num_sample)}
    _, scores = Meteor_scorer.compute_score(gts, res)

    return scores


def expand_reward(num_box, pos_idx, reward=None):
    output = np.zeros(num_box, dtype=np.float32)
    for i, idx in enumerate(pos_idx):
        if reward is None:
            output[idx] = 1.0
        else:
            output[idx] = reward[i]

    return output
