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
        pass

    def predict(self, xts):
        pass

    def predict(self,xts):
        # Calcola la distanza euclidea tra ciascuna immagine di test e tutti i centroidi
        # usando pairwise_distances di sklearn
        # Seleziona metric='euclidean' per calcolare la distanza euclidea
        distances = euclidean_distances(xts, self.centroids)
        
        # Per ciascuna immagine di test, si seleziona la classe con la distanza minore
        predictions = np.argmin(distances, axis=1)
        
        # Calcola l'accuratezza del modello
        accuracy = np.mean(predictions == self._class_labels) * 100
        
        print(f"Accuratezza: {accuracy:.2f}%")
        
        return predictions, accuracy