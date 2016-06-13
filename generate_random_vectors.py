import sys

from intrinsic_eval import word_level_datasets

sys.path.append('.')
import logging
from vector_utils import DenseVectors

import numpy as np
import pandas as pd

def get_all_words(**kwargs):
    res = set()
    for _, df in word_level_datasets():
        res = res.union(set(df['w1'])).union(set(df['w2']))
    return res


if __name__ == '__main__':

    """
    Generates a random vector for each NP in all labelled corpora
    """
    DIMENSIONALITY = 100
    OUT_PATH = 'vectors/random_vectors.hdf'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    np.random.seed(0)
    feats = ['rand%d' % i for i in range(DIMENSIONALITY)]
    words = list(get_all_words(include_unigrams=True))
    vectors = np.random.random((len(words), DIMENSIONALITY))

    v = DenseVectors(pd.DataFrame(vectors, index=words, columns=feats))
    v.to_hdf(OUT_PATH)