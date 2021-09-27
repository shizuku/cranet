import pickle
from typing import Any
import numpy as np


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def random_initializer(shape):
    return 0.1 * np.random.randn(*shape)
