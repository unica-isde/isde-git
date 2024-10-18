import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        """
        Compute the average centroids for each class

        Parameters
        ----------
        xtr: training data
        ytr: training labels

        Returns
        -------
        self: trained NMC classifier
        """

        n_dimensions = xtr.shape[1]
        n_classes = np.unique(ytr).size
        self._centroids = np.zeros(shape=(n_classes, n_dimensions))
        for k in range(n_classes):
            # extract images from one class and then average along dim 0
            self._centroids[k, :] = np.mean(xtr[ytr == k, :], axis=0)
        return self

    def predict(self, xts):
        """
        Brand new docstring

        Parameters
        ----------
        xts

        Returns
        -------

        """
        scores = self.decision_function(xts)
        ypred = np.argmax(scores, axis=1)
        return ypred



    def load_mnist(self):
        data = pd.read_csv("data/mnist_data.csv")
        data = np.array(data)

        y = data[:, 0]
        x = data[:, 1:] / 255

        return x,y
