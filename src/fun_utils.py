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

def fit(x_train, y_train):
    """
    Fit a model to the training data

    """
    num_classes = np.unique(y_train).size
    num_features = x_train.shape[1]

    centroids = np.zeros(shape=(num_classes,num_features))
    for i in range(num_classes):
        centroids[i] = np.mean(x_train[y_train == i], axis=0)
        return centroids
    