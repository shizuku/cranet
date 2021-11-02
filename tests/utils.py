import numpy as np


def np_feq(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-15) -> bool:
    return (np.abs(a - b) < epsilon).all()
