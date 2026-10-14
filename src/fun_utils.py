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
    idx = np.array(range(x.shape[0]))
    np.random.shuffle(idx)
    ntr = int(tr_fraction * x.shape[0])

    xtr = x[idx[0:ntr], :]
    ytr = y[idx[0:ntr]]

    xts = x[idx[ntr:], :]
    yts = y[idx[ntr:]]

    return xtr, ytr, xts, yts
