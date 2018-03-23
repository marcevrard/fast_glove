#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
Fast GloVe (PyTorch implementation)
===================================

Synopsis
--------
    examples:
    `````````
        ./fast_glove.py -tex --corpus-fpath ../word2vec_data/data_no_unk_tag.txt

Authors
-------
* Matthieu Bizien   (<https://github.com/MatthieuBizien>)
* Marc Evrard       (<marc.evrard@gmail.com>)
'''
import argparse
import json
import logging
import os
import re
from collections import Counter

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

USE_CUDA = torch.cuda.is_available()
CONFIG_FPATH = 'Config/config.json'
SEED = 0


def load_config(name):
    basepath, ext = os.path.splitext(CONFIG_FPATH)
    fpath = f'{basepath}_{name}{ext}'

    logging.info(f"Config file used: {fpath}")

    with open(fpath) as f:
        return json.load(f)


def cuda(x):
    if USE_CUDA:
        return x.cuda()
    return x


class WordIndexer:
    '''Transform g a dataset of text to a list of index of words.
    Not memory optimized for big datasets'''

    def __init__(self, occur_min=1, win_r=1, oov_word='OOV'):
        self.oov_word = oov_word
        self.win_r = win_r
        self.occur_min = occur_min
        self.word2index = {oov_word: 0}
        self.words = [oov_word]
        self.word_occurrences = {}
        self.words_re = re.compile(r'\b[a-zA-Z]{2,}\b')

    def _get_or_set_word_to_index(self, word):
        try:
            return self.word2index[word]
        except KeyError:
            idx = len(self.word2index)
            self.word2index[word] = idx
            self.words.append(word)
            return idx

    @property
    def words_num(self):
        return len(self.word2index)

    def fit_transform(self, texts):
        words_l = [list(self.words_re.findall(sentence.lower()))
                   for sentence in texts]
        word_occurrences = Counter(word for words in words_l for word in words)

        self.word_occurrences = {
            word: occurs_num
            for word, occurs_num in word_occurrences.items()
            if occurs_num >= self.occur_min}

        oov_index = 0
        return [[self._get_or_set_word_to_index(word)
                 if word in self.word_occurrences else oov_index
                 for word in words]
                for words in words_l]

    def _get_ngrams(self, indices):
        for i, index_l in enumerate(indices):
            window = indices[i + 1 : i + self.win_r + 1]
            for distance, index_r in enumerate(window):
                yield index_l, index_r, distance + 1

    def get_comatrix(self, data):
        comatrix = Counter()
        z = 0
        for indices in data:
            ngrams_l = self._get_ngrams(indices)
            for index_l, index_r, distance in ngrams_l:
                comatrix[(index_l, index_r)] += 1. / distance
                z += 1
        return zip(*[(left, right, x) for (left, right), x in comatrix.items()])


class GloveDataset(Dataset):
    def __len__(self):
        return self.obs_num

    def __getitem__(self, index):
        raise NotImplementedError()

    def __init__(self, texts, config):
        self.config = cfg = config
        torch.manual_seed(SEED)

        self.indexer = WordIndexer(win_r=cfg['win_r'],
                                   occur_min=cfg['occur_min'])
        data = self.indexer.fit_transform(texts)
        left, right, occurs_num = self.indexer.get_comatrix(data)
        occurs_num = np.array(occurs_num)
        self.obs_num = len(left)

        # We create the variables
        self.words_l = cuda(torch.LongTensor(left))
        self.words_r = cuda(torch.LongTensor(right))

        self.weights = np.minimum((occurs_num / cfg['x_max'])**cfg['alpha'], 1)
        self.weights = Variable(cuda(torch.FloatTensor(self.weights)))
        self.y = Variable(cuda(torch.FloatTensor(np.log(occurs_num))))

        # We create the embeddings and biases
        words_num = self.indexer.words_num
        vecs_l = cuda(torch.randn((words_num, cfg['emb_dim'])) * cfg['std_base'])   # pylint: disable=no-member
        vecs_r = cuda(torch.randn((words_num, cfg['emb_dim'])) * cfg['std_base'])   # pylint: disable=no-member
        biases_l = cuda(torch.randn((words_num,)) * cfg['std_base'])                # pylint: disable=no-member
        biases_r = cuda(torch.randn((words_num,)) * cfg['std_base'])                # pylint: disable=no-member
        self.params_all = [Variable(e, requires_grad=True)
                           for e in (vecs_l, vecs_r, biases_l, biases_r)]
        self.vecs_l, self.vecs_r, self.biases_l, self.biases_r = self.params_all


def gen_batchs(data, config):
    '''Batch sampling function'''
    cfg = config
    # TODO: replace by cuda(...)
    indices = torch.randperm(len(data))                                 # pylint: disable=no-member
    if USE_CUDA:
        indices = indices.cuda()
    for idx in range(0, len(data) - cfg['batch_size'] + 1, cfg['batch_size']):
        sample = indices[idx:idx + cfg['batch_size']]
        words_l, words_r = data.words_l[sample], data.words_r[sample]
        vecs_l = data.vecs_l[words_l]
        vecs_r = data.vecs_r[words_r]
        bias_l = data.biases_l[words_l]
        bias_r = data.biases_r[words_r]
        weight = data.weights[sample]
        y = data.y[sample]
        yield weight, vecs_l, vecs_r, y, bias_l, bias_r


def get_loss(weight, vecs_l, vecs_r, log_covals, bias_l, bias_r):
    sim = (vecs_l * vecs_r).sum(1).view(-1)
    x = (sim + bias_l + bias_r - log_covals) ** 2
    loss = torch.mul(x, weight)
    return loss.mean()


def train_model(data: GloveDataset, config):
    optimizer = torch.optim.Adam(data.params_all, weight_decay=1e-8)
    optimizer.zero_grad()
    for epoch in tqdm(range(config['epochs_num'])):
        logging.info(f"Start epoch {epoch}")
        batches_num = int(len(data) / config['batch_size'])
        avg_loss = 0.0
        batch_n = int(len(data) / config['batch_size'])
        for batch in tqdm(gen_batchs(data, config), total=batch_n, mininterval=1):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            avg_loss += loss.data[0] / batches_num
            loss.backward()
            optimizer.step()
        logging.info(f"Average loss for epoch {epoch + 1}: {avg_loss: .5f}")


def get_args(args=None):     # Add possibility to manually insert args at runtime (e.g. for ipynb)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--corpus-type', choices=['big', 'toy', 'wn'], default='big',
                        help='Training dataset name.')

    return parser.parse_args(args)


def main(argp):

    config = load_config(name=argp.corpus_type)

    logging.info("Fetching data")
    newsgroup = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

    logging.info("Build dataset")
    glove_data = GloveDataset(newsgroup.data, config)
    logging.info(f"#Words: {glove_data.indexer.words_num}")
    logging.info(f"#Ngrams: {len(glove_data)}")

    logging.info("Start training")
    train_model(glove_data, config)


if __name__ == '__main__':
    main(get_args())
