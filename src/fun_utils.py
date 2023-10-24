from pandas import read_csv
import numpy as np
from sklearn.metrics import pairwise_distances


def predict(self, xts):
    """
    Compute predictions on test data.

    Parameters
    ----------
    xts
        Test data

    Returns
    -------
    Predicted labels for the test data.

    """
    if self._centroids is None:
        raise ValueError("Centroids not set. Run fit(x,y) first!")

    dist = pairwise_distances(xts, self._centroids)
    ypred = np.argmin(dist, axis=1)
    return ypred