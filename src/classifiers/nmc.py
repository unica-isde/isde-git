import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


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

    def fit(self, Xtr, ytr):
        self._class_labels = np.unique(ytr)
        num_classes = self._class_labels.size
        self._centroids = np.zeros(shape=(num_classes, Xtr.shape[1]))
        for k in range(num_classes):
            xk = Xtr[ytr == self._class_labels[k], :]
            self._centroids[k, :] = np.mean(xk, axis=0)

    def predict(self, xts):
        pass
