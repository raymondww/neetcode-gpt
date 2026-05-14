import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        transform = list(map(lambda x: np.e**x,z-max(z)))
        total = sum(transform)
        return list(map(lambda x: round(x/total,4),transform))