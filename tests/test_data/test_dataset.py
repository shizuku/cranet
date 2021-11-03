import unittest
from pathlib import Path
from itertools import accumulate

from src.cranet.data import random_split
import cravision


class TestDataset(unittest.TestCase):
    def test_random_split(self):
        from numpy.random import default_rng
        lengths = [3, 4, 5]
        rng = default_rng()
        indices = rng.permutation(sum(lengths)).tolist()
        print()
        print(f"indices:{indices}")
        print(f"accumulate(indices):{list(accumulate(indices))}")
        print()
        [print(indices[offset - length: offset]) for offset, length in zip(accumulate(lengths), lengths)]

    def test_random_split_1(self):
        HOME = Path.home()
        DATA_DIR = HOME / "Downloads" / "dataset"
        cradataset = cravision.datasets.SVHN(root=DATA_DIR, transform=cravision.transforms.ToTensor())

        VAL_SIZE = 5000  # validation data size
        TRAIN_SIZE = len(cradataset) - VAL_SIZE  # training data size

        train_ds, val_ds = random_split(cradataset, [TRAIN_SIZE, VAL_SIZE])
        print()
        print(len(train_ds), len(val_ds))
