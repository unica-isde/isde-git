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
    x_samples = x.shape[0]
    tr_samples = round(x_samples * tr_fraction)
    x_idx = np.array(range(0, x_samples))
    np.random.shuffle(x_idx)

    tr_idx = x_idx[0:tr_samples]
    ts_idx = x_idx[tr_samples:]

    xtr = x[tr_idx, :]
    ytr = y[tr_idx]
    xts = x[ts_idx, :]
    yts = y[ts_idx]

    return xtr, ytr, xts, yts
