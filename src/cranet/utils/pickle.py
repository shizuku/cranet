import pickle

from typing import Any


def save(obj: Any, f: Any):
    if isinstance(f, str):
        with open(f, 'wb') as file:
            pickle.dump(obj, file)
    else:
        raise TypeError


def load(f: Any) -> Any:
    if isinstance(f, str):
        with open(f, 'rb') as file:
            return pickle.load(file)
    else:
        raise TypeError
