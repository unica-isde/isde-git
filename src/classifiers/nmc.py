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

    def fit(self, xtr, ytr):
        num_classes = np.unique(ytr).size
        # the number of colums of x
        num_features = xtr.shape[1]
        self._centroids = np.zeros(shape=(num_classes, num_features))

        for k in range(num_classes):
            xk = xtr[ytr == k, :]
            self._centroids[k, :] = np.mean(xk, axis=0)
        return self
    def predict(self, xts):
        if self._centroids is None:
            raise ValueError("train classifier ")
        dist = euclidean_distances(xts, self.centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred