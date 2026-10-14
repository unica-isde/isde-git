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
    pass

def load_mnist(csv_filename):
    """Load the MNIST dataset.
    input:
        csv_filename: string with the path to the dataset to load
    output:
        X: set of images (numpy array)
        y: labels (numpy array)
    """
    data = read_csv(csv_filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y