import numpy as np
from os import path
import unittest

from fun_utils import load_data, split_data
from classifiers import NMC


class TestNMC(unittest.TestCase):

    def setUp(self):
        # Loading MNIST dataset
        mnist_path = path.join(
            path.dirname(__file__), '..', '../data', 'mnist_data.csv')
        x, y = load_data(filename=mnist_path)
        # Rescale data in 0-1
        self.x = x / 255
        self.y = y
        self.clf = NMC()

    def test_split_data(self):
        xtr, ytr, xts, yts = split_data(self.x, self.y, tr_fraction=0.5)

        assert type(xtr) == np.ndarray
        assert type(ytr) == np.ndarray
        assert type(xts) == np.ndarray
        assert type(yts) == np.ndarray

        assert xtr.ndim == 2
        assert ytr.ndim == 1
        assert xts.ndim == 2
        assert yts.ndim == 1

        tr_size = int(self.x.shape[0] * 0.5)
        ts_size = self.x.shape[0] - xtr.shape[0]

        assert xtr.shape[0] == tr_size
        assert ytr.size == tr_size
        assert xts.shape[0] == ts_size
        assert yts.size == ts_size

        assert xtr.shape[1] == self.x.shape[1]
        assert xts.shape[1] == self.x.shape[1]

    def test_fit(self):
        self.clf.fit(self.x, self.y)
        assert self.clf.centroids is not None
        assert self.clf.centroids.shape[0] == np.unique(self.y).size
        assert self.clf.centroids.shape[1] == self.x.shape[1]

    def test_predict(self):
        self.clf.fit(self.x, self.y)
        yc = self.clf.predict(self.x)
        assert yc.ndim == 1
        assert yc.shape[0] == self.x.shape[0]
