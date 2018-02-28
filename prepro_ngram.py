import json
import argparse
from six.moves import cPickle
from collections import defaultdict


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  # lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def build_dict(imgs, wtoi, params):
    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        if (params['split'] == img['split']) or \
                (params['split'] == 'train' and img['split'] == 'restval') or \
                (params['split'] == 'all'):
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                tmp_tokens = sent['tokens'] + ['<END>']
                tmp_tokens = [_ if _ in wtoi else '<UNK>' for _ in tmp_tokens]
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    word_to_idx = json.load(open(params['dict_json'], 'r'))['word_to_idx']
    itow = {i: w for w, i in word_to_idx.items()}
    wtoi = {w: i for i, w in itow.items()}
    imgs = imgs['images']

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, params)

    cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len},
                 open(params['output_pkl']+'-words.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len},
                 open(params['output_pkl']+'-idxs.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='data/coco/dataset_coco.json', help='input json file with dataset')
    parser.add_argument('--dict_json', default='data/coco/coco_labels.json', help='input json file with labels')
    parser.add_argument('--output_pkl', default='data/coco/coco-train', help='output pickle file')
    parser.add_argument('--split', default='train', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    main(params)
