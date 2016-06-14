# -*- coding: utf-8 -*-
import argparse
import os
import sys
import math
from os.path import join
import logging
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import RULE_DISCARD, RULE_KEEP
from generate_random_vectors import get_all_words
from vector_utils import DenseVectors

# sys.path.append('.')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# training parameters
MIN_COUNT = 50
WORKERS = 10
ALL_WORDS = get_all_words()


def mkdirs_if_not_exists(dir):
    """
    Creates a directory (and all intermediate directories) if it doesn't exists.
    Behaves like mkdir -p, and is prone to race conditions
    Source: http://stackoverflow.com/q/273192/419338
    :param dir:
    :return:
    """
    if not (os.path.exists(dir) and os.path.isdir(dir)):
        os.makedirs(dir)


class MySentences(object):
    def __init__(self, dirname, file_percentage):
        self.dirname = dirname
        self.limit = file_percentage / 100

        files = [x for x in sorted(os.listdir(self.dirname)) if not x.startswith('.')]
        count = math.ceil(self.limit * len(files))
        # always take the same files for the first repetition so we can plot a learning curve that shows the
        # effect of adding a bit of extra data, e.g. going from 50% to 60% of corpus.
        self.files = files[:count]
        logging.info('Will use the following %d files for training\n %s', len(self.files), self.files)

    def __iter__(self):
        for fname in self.files:
            with open(join(self.dirname, fname)) as infile:
                for line in infile:
                    # yield gensim.utils.tokenize(line, lower=True)
                    if isinstance(line, bytes):
                        line = line.decode()
                    res = [w.lower() for w in line.split()]
                    if len(res) > 8:
                        # ignore short sentences, they are probably noise
                        yield res


def compute_and_write_vectors(input_dir, output_file, percent):
    mkdirs_if_not_exists(os.path.dirname(output_file))

    logging.info('Training word2vec on %d percent of %s', percent, input_dir)
    sentences = MySentences(input_dir, percent)

    def trimmer(word, count, min_count):
        if word in ALL_WORDS and count >= min_count:
            return RULE_KEEP
        else:
            return RULE_DISCARD

    model = Word2Vec(sentences, workers=WORKERS, min_count=MIN_COUNT,
                     seed=0, trim_rule=trimmer)

    vocab = list(sorted(model.vocab.keys()))
    dims = len(model[next(iter(vocab))])  # vector dimensionality
    dimension_names = ['f%02d' % i for i in range(dims)]
    matrix = model[vocab]

    vectors = DenseVectors(pd.DataFrame(matrix, index=vocab, columns=dimension_names))
    vectors.to_hdf(output_file)

def get_args_from_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-file', required=True)
    # percent of files to use. SGE makes it easy for this to be 1, 2, ...
    parser.add_argument('--percent', default=100, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_from_cmd_line()
    logging.info('Params are: %r', args)
    compute_and_write_vectors(args.input_dir, args.output_file, args.percent)
