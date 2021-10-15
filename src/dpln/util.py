import numpy as np

import pickle
from typing import Any


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
