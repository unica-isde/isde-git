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
    """
    Split the data x, y into two random subsets

    """
    samples = x.shape[0]
    idx = list(range(0, samples))
    np.random.shuffle(idx)
    n_tr = int(tr_fraction * samples)

    idx_tr = idx[:n_tr]
    idx_ts = idx[n_tr:]

    xtr = x[idx_tr, :]
    ytr = y[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]

    return xtr, ytr, xts, yts
