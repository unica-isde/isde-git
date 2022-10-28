from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    num_samples = y.size
    num_tr = int(tr_fraction * num_samples)
    num_ts = num_samples - num_tr
    tr_idx = np.zeros(shape=(num_samples,))
    tr_idx[0:num_tr] = 1

    np.random.shuffle(tr_idx)
    ytr = y[tr_idx == 1]
    xtr = x[tr_idx == 1, :]

    yts = y[tr_idx == 0]
    xts = x[tr_idx == 0, :]

    return xtr, ytr, xts, yts




