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

def load_mnist_data(filename, n_samples=None):
	"""This function returns MNIST handwritten digits and labels as ndarrays."""
	data = pd.read_csv(filename)
	data = np.array(data)  # cast pandas dataframe to numpy array
	if n_samples is not None:  # only returning the first n_samples
		data = data[:n_samples, :]
	y = data[:,0]
	x = data[:,1:] / 255.0
	return x, y

def split_data(x, y, tr_fraction=0.5):
    n_samples = x.shape[0]

    n_tr = int(tr_fraction*n_samples)
    n_ts = n_samples - n_tr

    # create a vector of indices from 0 to 999 = [0, 1, 2, ... n_samples-1]
    idx = np.linspace(0, n_samples, num=n_samples, endpoint=False, dtype='int')
    np.random.shuffle(idx)  # shuffling the elements of idx (in-place)

    tr_idx = idx[:n_tr]  # extract the subset of training indices
    ts_idx = idx[n_tr:]  # extract the subset of test indices

    # check coherence with the number of extracted elements
    assert(n_tr==tr_idx.size)
    assert(n_ts==ts_idx.size)

    xtr = x[tr_idx,:]
    ytr = y[tr_idx]
    xts = x[ts_idx,:]
    yts = y[ts_idx]
    return xtr, ytr, xts, yts
    pass

