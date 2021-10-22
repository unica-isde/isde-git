from pandas import read_csv
import numpy as np


def load_data(filename, n_samples=None):
    """
    Load data from a csv file

    Parameters
    ----------
    n_samples : integer
        number of samples.
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
    if n_samples is not None:
        z = z[:n_samples, :]
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    pass
