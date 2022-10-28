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
    n_tr = int(tr_fraction * num_samples)
    n_ts = num_samples - n_tr

    idx = np.arange(num_samples)

    np.random.shuffle(idx)
    idx_tr = idx[:n_tr]
    idx_ts = idx[n_tr:]
    assert (n_ts == idx_ts.size)
    xtr = x[idx_tr, :]
    ytr = y[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]
    return xtr, ytr, xts, yts

