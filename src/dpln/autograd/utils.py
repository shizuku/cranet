import numpy as np


def invert_permutation(permutation):
    if type(permutation) is tuple:
        permutation = list(permutation)
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv
