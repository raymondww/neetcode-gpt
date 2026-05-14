import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        # ln(x) = np.log(x)
        total = 0
        for a,b in zip(y_true,y_pred + 1e-7):
            total += a*np.log(b)+(1-a)*np.log(1-b)
        return round(-(1/len(y_true)) * total,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        total = 0
        for a,b in zip(y_true,y_pred + 1e-7):
            class_total = 0
            for c,d in zip(a,b):
                class_total += c*np.log(d)
            total += class_total
        return round(-(1/len(y_true)) * total,4)

