import numpy as np
from collections import Counter


class KNN:
    """
    K-Nearest Neighbors classifier
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x: np.ndarray):
        distances = [
            self._euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]

        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
