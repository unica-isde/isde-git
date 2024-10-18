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
    """Split data to create a random tr-ts partition."""
    n, d = x.shape

    # check if y and x have a consistent no. of samples and labels
    n1 = y.size
    assert (n == n1)

    n_tr = int(np.round(n * tr_fraction))

    idx = np.array(range(0, n))  # 0, 1, 2, ..., n-1
    np.random.shuffle(idx)
    idx_tr = idx[0:n_tr]
    idx_ts = idx[n_tr:n]

    xtr = x[idx_tr, :]
    ytr = y[idx_tr]
    xts = x[idx_ts, :]
    yts = y[idx_ts]
    return xtr,ytr,xts,yts
