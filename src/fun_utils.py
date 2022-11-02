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
    x = z[:, 1:]
    return x, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """

    "Split the data sizes into two, and randomly choosing values for both the test and training subsets"
    n_sample = x.size(0)
    n_tr = int(n_sample * tr_fraction)

    ind = np.linespace(1, n_sample, num=n_sample, endpoint=False, dtype='int')
    np.random.shuffle(ind)

    "Extraction of the two subsets' indexes, tr_ind = training, ts_ind = testing indices"
    tr_ind = ind[:n_tr]
    ts_ind = ind[n_tr:]

    xtr = x[tr_ind:, :]
    xts = x[ts_ind, :]
    ytr = y[tr_ind]
    yts = y[ts_ind]

    return xtr, ytr, xts, yts
    pass
