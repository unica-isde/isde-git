from pandas import read_csv
import numpy as np


def load_mnist(filename):
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

    num_samples = y.size
    n_tr = int(num_samples * tr_fraction)
    idx = np.array(range(0, num_samples))
    np.random.shuffle(idx)  # shuffle indices
    tr_idx = idx[0:n_tr]
    ts_idx = idx[n_tr:]
    Xtr = x[tr_idx, :]
    ytr = y[tr_idx]
    Xts = x[ts_idx, :]
    yts = y[ts_idx]
    return Xtr,ytr,Xts,yts

