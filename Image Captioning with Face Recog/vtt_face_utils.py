# -*- coding: utf-8 -*-
import numpy as np
import re


class NameReplacer(object):
    def __init__(self, idx_to_word, score_thr=0.1):
        base_words = ['rachel', 'monica', 'phoebe', 'ross', 'chandler','joey',
                      'carole', 'angela', 'bob', 'janice', 'man', 'woman', 'person']
        article_words = ['a', 'the']

        self.idx_to_word = idx_to_word
        self.score_thr = score_thr

        self.base_words = base_words
        self.article_words = article_words

    # convert index(int) to captions(characters)
    def _convert_to_caption(self, captions):
        if captions.ndim == 1:
            len_cap = captions.shape[0]
            num_cap = 1
        else:
            num_cap, len_cap = captions.shape

        decoded = []
        for i in range(num_cap):
            words = []
            for t in range(len_cap):
                if captions.ndim == 1:
                    word = self.idx_to_word[captions[t]]
                else:
                    word = self.idx_to_word[captions[i, t]]

                if word == '<END>':
                    words.append('.')
                    break

                if word != '<NULL>':
                    words.append(word)
            decoded.append(words)

        return decoded

    # calc score per person using alpha map
    # using only part with the largest alpha value
    @staticmethod
    def _calc_person_score_multialpha(alpha, start, end, bbox, person_name, person_prob, person_bbox, person_flag):
        scores = []

        for p in range(len(person_name)):
            score_person = 0.0
            iou_person = 0.0
            if person_flag[p] == 0:
                for i in range(start, end+1):
                    alpha_max_idx = np.argmax(alpha[i, :])
                    obj_box = [bbox[alpha_max_idx, 0], bbox[alpha_max_idx, 1],
                               bbox[alpha_max_idx, 2], bbox[alpha_max_idx, 3]]
                    iou = object_overlap(obj_box, person_bbox[p])
                    score = iou * person_prob[p][0]
                    if score > score_person:
                        score_person = score
                        iou_person = iou
                    if iou>1:
                        print()
            scores.append(score_person)
            print("\n\tPerson Name: %s" % person_name[p])
            print("\tPerson Prob: %.4f" % person_prob[p][0])
            print("\tPerson iou: %.4f" % iou_person)
        print("===== END")

        # descending order with score
        idx = sorted(range(len(scores)), key=scores.__getitem__)
        idx.reverse()
        scores.sort(reverse=True)

        return idx, scores

    # search for replacement part
    # personal noun, article word, ...
    def _search_for_replacement(self, caption):
        # replace index list
        # 0 : None, 1 : base (singular), 2 : article
        replace_idx = np.zeros(len(caption), dtype=np.int8)
        replace_word = []

        for i in range(len(caption)-1, -1, -1):
            flag = 0

            # is base word?
            for w in self.base_words:
                if str(caption[i]).find(str(w)) > -1:
                    flag = 1
                    replace_word.append(str(w))
                    break

            # is article word?
            if flag == 0:
                for w in self.article_words:
                    if str(caption[i]) == str(w):
                        flag = 2
                        replace_word.append(str(w))
                        break

            if flag == 0:
                replace_word.append('')
            replace_idx[i] = flag

        replace_word.reverse()
        return replace_idx, replace_word

    # replace word to person
    def _replace_word(self, score_idx, scores, person_name, person_flag):
        words = []
        if len(scores) >= 1 and scores[0] > self.score_thr:
            words.append(person_name[score_idx[0]])
            person_flag[score_idx[0]] = 1

        return ' '.join(words), person_flag

    # replace caption
    def _replace_cap(self, caption, replace_idxs, alphas, bbox, person_name, person_prob, person_bbox):
        start_idx = 0
        processing = False
        new_cap = []
        person_flag = np.zeros(len(person_name), dtype=np.int8)

        # based new(replaced) caption
        for i in range(len(caption)):
            new_cap.append(caption[i])

        for i in range(len(caption)):
            if replace_idxs[i] == 2:
                start_idx = i
                processing = True
            # replace word
            elif replace_idxs[i] == 1:
                if processing is False:
                    start_idx = i
                score_idx, scores = self._calc_person_score_multialpha(alphas, start_idx, i, bbox, person_name,
                                                                       person_prob, person_bbox, person_flag)
                word, person_flag = self._replace_word(score_idx, scores, person_name, person_flag)
                if len(word) > 0:
                    for j in range(start_idx, i):
                        new_cap[j] = '<BLK>'
                    new_cap[i] = word
                processing = False
            else:
                processing = False

        return new_cap

    def name_replace_caps(self, captions, alphas, bbox, person_name, person_prob, person_bbox):
        caps = self._convert_to_caption(captions=captions)
        len_cap = len(caps)

        decoded = []
        words = []
        for i in range(len_cap):
            replace_idxs, replace_words = self._search_for_replacement(caps[i])
            replace_cap = self._replace_cap(caps[i], replace_idxs,
                                            alphas[i, :, :], bbox, person_name, person_prob, person_bbox)
            words.append(replace_cap)
            del_blk_caps = ' '.join(replace_cap).replace('<BLK>', '')
            decoded.append(del_blk_caps)
        return decoded, words


def object_overlap(det_box, src_box):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(src_box[0], det_box[0])
    y1 = max(src_box[1], det_box[1])
    x2 = min(src_box[2], det_box[2])
    y2 = min(src_box[3], det_box[3])

    # compute the area of intersection rectangle
    inter_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if (x2 - x1 + 1) <= 0:
        return 0
    elif (y2 - y1 + 1) <= 0:
        return 0

    target_box_area = (src_box[2] - src_box[0] + 1) * (src_box[3] - src_box[1] + 1)
    iou = inter_area / float(target_box_area)

    return iou
