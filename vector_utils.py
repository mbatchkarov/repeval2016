import logging
import os
import numpy as np
import pandas as pd


class DenseVectors(object):
    """
    A thin wrapper around a pandas DataFrame for storing dense word vectors.
    This is a stripped-down version of discoutils.thesaurus_loader
    """

    def __init__(self, df, noise=False):
        self.df = df

        self.matrix, self.columns, self.row_names = self.df.values, self.df.columns, self.df.index.values
        if noise:
            logging.info('Adding uniform noise U[-{0}, +{0}] to vectors'.format(noise))
            self.matrix += np.random.uniform(-noise, noise, self.matrix.shape)
        self.name2row = {feature: i for (i, feature) in enumerate(self.row_names)}

    def get_vector(self, item):
        if item not in self.name2row:
            return None
        return self.df.ix[item].values

    def __getitem__(self, item):
        return zip(self.columns, self.get_vector(item))

    def keys(self):
        return self.df.index

    def __len__(self):
        return len(self.row_names)

    def __str__(self):
        return '[Dense vectors of shape {}]'.format(self.df.shape)

    @classmethod
    def from_hdf(cls, hdf_file,
                 row_filter=lambda _: True,
                 **kwargs):
        df = pd.read_hdf(hdf_file, 'matrix')
        logging.info('Found a DF of shape %r in HDF file %s', df.shape, hdf_file)
        # pytables doesn't like unicode values and replaces them with an empty string.
        # pandas doesn't like duplicate values in index
        # remove these, we don't want to work with them anyway
        df = df[df.index != '']
        row_filter_mask = [row_filter(f) for f in df.index]
        df = df[row_filter_mask]
        logging.info('Dropped non-ascii rows and applied row filter. Shape is now %r', df.shape)
        return DenseVectors(df, **kwargs)

    def to_hdf(self, path):
        matrix = self.matrix
        row_index = self.row_names
        column_index = self.columns

        logging.info('Writing vectors of shape %r to %s', matrix.shape, path)
        if isinstance(row_index, dict):
            # row_index is a dict, let's make it into a list
            ri = list(range(len(row_index)))  # mega inefficient, but numpy str arrays confuse me
            for phrase, idx in row_index.items():
                try:
                    str(phrase).encode('ascii')
                    ri[idx] = str(phrase)
                except UnicodeEncodeError as e:
                    # pandas doesnt like non-ascii keys in index; mark such phrases for removal
                    ri[idx] = 'THIS_IS_NOT_RIGHT_%d' % idx
        else:
            ri = list(map(str, row_index))
        old_shape = matrix.shape
        # remove phrases that arent ascii-only
        to_keep = np.array([False if str(x).startswith('THIS_IS_NOT_RIGHT_') else True for x in ri])
        matrix = matrix[to_keep, :]
        ri = np.array(ri)[to_keep]
        if old_shape != matrix.shape:
            logging.info('Removing non-ascii phrases. Matrix shape was %r, is now %r', old_shape, matrix.shape)

        df = pd.DataFrame(matrix, index=ri, columns=map(str, column_index))
        if os.path.exists(path):
            # PyTables fails if the file exist, but is not and HDF store. Remove the file
            os.unlink(path)
        df.to_hdf(path, 'matrix', complevel=9, complib='zlib')

        return path

