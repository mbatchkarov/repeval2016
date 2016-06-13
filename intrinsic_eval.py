from itertools import chain
import sys
import logging

from vector_utils import DenseVectors
from joblib.parallel import delayed, Parallel
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

sys.path.append('.')
PATHS = ['vectors/word2vec-wiki-nopos-100perc.unigr.strings.rep0',
         'vectors/random_vectors.hdf']
NAMES = ['wtv-wiki-100', 'random']
NBOOT = 500


def _ws353():
    df = pd.read_csv('similarity-data/wordsim353/combined.csv',
                     names=['w1', 'w2', 'sim'])
    df.w1 = df.w1.map(str.lower)
    df.w2 = df.w2.map(str.lower)
    return df


def _mc():
    return pd.read_csv('similarity-data/miller-charles.txt',
                       names=['w1', 'w2', 'sim'], sep='\t')


def _rg():
    return pd.read_csv('similarity-data/rub-gooden.txt',
                       names=['w1', 'w2', 'sim'], sep='\t')


def _men():
    df = pd.read_csv('similarity-data/MEN/MEN_dataset_lemma_form_full',
                     names=['w1', 'w2', 'sim'], sep=' ')

    def _remove_pos_tag(word):
        return word[:-2] # remove PoS tag

    df.w1 = df.w1.map(_remove_pos_tag)
    df.w2 = df.w2.map(_remove_pos_tag)
    return df


def _simlex999():
    df = pd.read_csv('similarity-data/SimLex-999/SimLex-999.txt',
                     names='word1,word2,POS,SimLex999,conc(w1),conc(w2),concQ,Assoc(USF),'
                           'SimAssoc333,SD(SimLex)'.split(','),
                     sep='\t', skiprows=1)
    df = df[['word1', 'word2', 'SimLex999']]
    df.columns = ['w1', 'w2', 'sim']
    return df


def word_level_datasets():
    yield 'ws353', _ws353()
    yield 'mc', _mc()
    yield 'rg', _rg()
    yield 'men', _men()
    yield 'simlex', _simlex999()


def cosine_similarity(v1, v2):
    return 1 - cosine_distance(v1, v2)


def _intrinsic_eval_words(vectors_path, intrinsic_dataframe, noise=0):
    # if 'random' in vectors_path and noise < 0.01:
    #     return [] # save some time

    all_words = set(intrinsic_dataframe.w1).union(set(intrinsic_dataframe.w2))
    v = DenseVectors.from_hdf(vectors_path, noise=noise, row_filter=lambda w: w in all_words)
    model_sims, human_sims = [], []
    missing = 0
    for w1, w2, human in zip(intrinsic_dataframe.w1,
                             intrinsic_dataframe.w2,
                             intrinsic_dataframe.sim):
        v1, v2 = v.get_vector(w1), v.get_vector(w2)
        if v1 is not None and v2 is not None:
            model_sims.append(cosine_similarity(v1, v2))
            human_sims.append(human)
        else:
            missing += 1

    # where model failed to answer insert something at random
    model_sims_w_rand = model_sims + list(np.random.uniform(min(model_sims), max(model_sims), missing))
    human_sims_w_rand = human_sims + list(np.random.uniform(min(human_sims), max(human_sims), missing))

    # sanity check: strict accuracy results must be lower than relaxed
    # some allowance for change- random guesses may slightly improve results
    relaxed, _ = spearmanr(model_sims, human_sims)
    strict, _ = spearmanr(model_sims_w_rand, human_sims_w_rand)
    if strict > relaxed:
        logging.warning('Strict correlation higher than relaxed: %.2f, %.2f', strict, relaxed)

    # bootstrap model_sims_w_rand CI for the data
    res = []
    for boot_i in range(NBOOT):
        idx = np.random.randint(0, len(model_sims), len(model_sims))
        relaxed, rel_pval = spearmanr(np.array(model_sims)[idx],
                                      np.array(human_sims)[idx])

        idx = np.random.randint(0, len(model_sims_w_rand), len(model_sims_w_rand))
        strict, str_pval = spearmanr(np.array(model_sims_w_rand)[idx],
                                     np.array(human_sims_w_rand)[idx])

        res.append([strict, relaxed, noise, rel_pval, str_pval,
                    missing / len(intrinsic_dataframe), boot_i])
    return res


def noise_eval():
    """
    Test: intrinsic eval on noise-corrupted vectors

    Add noise as usual, evaluated intrinsically.
    """
    noise_data = []
    for dname, df in word_level_datasets():
        for vname, path in zip(NAMES, PATHS):
            logging.info('starting %s %s', dname, vname)
            res = Parallel(n_jobs=-1)(delayed(_intrinsic_eval_words)(path, df, noise) \
                                      for noise in np.arange(0, 3.1, .2))
            for strict, relaxed, noise, rel_pval, str_pval, _, boot_i in chain.from_iterable(res):
                noise_data.append((vname, dname, noise, 'strict', strict, str_pval, boot_i))
                noise_data.append((vname, dname, noise, 'relaxed', relaxed, rel_pval, boot_i))
    noise_df = pd.DataFrame(noise_data,
                            columns=['vect', 'test', 'noise', 'kind', 'corr', 'pval', 'folds'])
    noise_df.to_csv('intrinsic_noise_word_level.csv')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(module)s.%(funcName)s (line %(lineno)d)\t"
                               "%(levelname)s : %(message)s")
    noise_eval()
    logging.info('Done')
