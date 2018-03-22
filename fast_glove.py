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

    def __init__(self, min_word_occurrences=1, right_window=1, oov_word='OOV'):
        self.oov_word = oov_word
        self.right_window = right_window
        self.min_word_occurrences = min_word_occurrences
        self.word_to_index = {oov_word: 0}
        self.index_to_word = [oov_word]
        self.word_occurrences = {}
        self.re_words = re.compile(r'\b[a-zA-Z]{2,}\b')

    def _get_or_set_word_to_index(self, word):
        try:
            return self.word_to_index[word]
        except KeyError:
            idx = len(self.word_to_index)
            self.word_to_index[word] = idx
            self.index_to_word.append(word)
            return idx

    @property
    def n_words(self):
        return len(self.word_to_index)

    def fit_transform(self, texts):
        l_words = [list(self.re_words.findall(sentence.lower()))
                   for sentence in texts]
        word_occurrences = Counter(word for words in l_words for word in words)

        self.word_occurrences = {
            word: n_occurrences
            for word, n_occurrences in word_occurrences.items()
            if n_occurrences >= self.min_word_occurrences}

        oov_index = 0
        return [[self._get_or_set_word_to_index(word)
                 if word in self.word_occurrences else oov_index
                 for word in words]
                for words in l_words]

    def _get_ngrams(self, indexes):
        for i, left_index in enumerate(indexes):
            window = indexes[i + 1 : i + self.right_window + 1]
            for distance, right_index in enumerate(window):
                yield left_index, right_index, distance + 1

    def get_comatrix(self, data):
        comatrix = Counter()
        z = 0
        for indexes in data:
            l_ngrams = self._get_ngrams(indexes)
            for left_index, right_index, distance in l_ngrams:
                comatrix[(left_index, right_index)] += 1. / distance
                z += 1
        return zip(*[(left, right, x) for (left, right), x in comatrix.items()])


class GloveDataset(Dataset):
    def __len__(self):
        return self.n_obs

    def __getitem__(self, index):
        raise NotImplementedError()

    def __init__(self, texts, config):
        self.config = cfg = config
        torch.manual_seed(SEED)

        self.indexer = WordIndexer(right_window=cfg['right_window'],
                                   min_word_occurrences=cfg['min_word_occurrences'])
        data = self.indexer.fit_transform(texts)
        left, right, n_occurrences = self.indexer.get_comatrix(data)
        n_occurrences = np.array(n_occurrences)
        self.n_obs = len(left)

        # We create the variables
        self.l_words = cuda(torch.LongTensor(left))
        self.r_words = cuda(torch.LongTensor(right))

        self.weights = np.minimum((n_occurrences / cfg['x_max'])**cfg['alpha'], 1)
        self.weights = Variable(cuda(torch.FloatTensor(self.weights)))
        self.y = Variable(cuda(torch.FloatTensor(np.log(n_occurrences))))

        # We create the embeddings and biases
        n_words = self.indexer.n_words
        l_vecs = cuda(torch.randn((n_words, cfg['n_embedding'])) * cfg['base_std'])     # pylint: disable=no-member
        r_vecs = cuda(torch.randn((n_words, cfg['n_embedding'])) * cfg['base_std'])     # pylint: disable=no-member
        l_biases = cuda(torch.randn((n_words,)) * cfg['base_std'])                      # pylint: disable=no-member
        r_biases = cuda(torch.randn((n_words,)) * cfg['base_std'])                      # pylint: disable=no-member
        self.all_params = [Variable(e, requires_grad=True)
                           for e in (l_vecs, r_vecs, l_biases, r_biases)]
        self.l_vecs, self.r_vecs, self.l_biases, self.r_biases = self.all_params


def gen_batchs(data, config):
    '''Batch sampling function'''
    cfg = config
    # TODO: replace by cuda(...)
    indices = torch.randperm(len(data))                                 # pylint: disable=no-member
    if USE_CUDA:
        indices = indices.cuda()
    for idx in range(0, len(data) - cfg['batch_size'] + 1, cfg['batch_size']):
        sample = indices[idx:idx + cfg['batch_size']]
        l_words, r_words = data.l_words[sample], data.r_words[sample]
        l_vecs = data.l_vecs[l_words]
        r_vecs = data.r_vecs[r_words]
        l_bias = data.l_biases[l_words]
        r_bias = data.r_biases[r_words]
        weight = data.weights[sample]
        y = data.y[sample]
        yield weight, l_vecs, r_vecs, y, l_bias, r_bias


def get_loss(weight, l_vecs, r_vecs, log_covals, l_bias, r_bias):
    sim = (l_vecs * r_vecs).sum(1).view(-1)
    x = (sim + l_bias + r_bias - log_covals) ** 2
    loss = torch.mul(x, weight)
    return loss.mean()


def train_model(data: GloveDataset, config):
    optimizer = torch.optim.Adam(data.all_params, weight_decay=1e-8)
    optimizer.zero_grad()
    for epoch in tqdm(range(config['num_epoch'])):
        logging.info(f"Start epoch {epoch}")
        num_batches = int(len(data) / config['batch_size'])
        avg_loss = 0.0
        n_batch = int(len(data) / config['batch_size'])
        for batch in tqdm(gen_batchs(data, config), total=n_batch, mininterval=1):
            optimizer.zero_grad()
            loss = get_loss(*batch)
            avg_loss += loss.data[0] / num_batches
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
    logging.info(f"#Words: {glove_data.indexer.n_words}")
    logging.info(f"#Ngrams: {len(glove_data)}")

    logging.info("Start training")
    train_model(glove_data, config)


if __name__ == '__main__':
    main(get_args())
