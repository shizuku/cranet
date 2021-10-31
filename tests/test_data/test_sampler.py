import unittest

from src.dpln.data import SequentialSampler, BatchSampler, RandomSampler
import numpy as np


class TestSequentialSampler(unittest.TestCase):
    def test_seq_0(self):
        print(list(SequentialSampler(np.random.rand(10))))

    def test_seq_1(self):
        print(list(SequentialSampler(np.random.rand(20))))


class TestRandomSampler(unittest.TestCase):
    def test_rand_0(self):
        print(list(RandomSampler(np.random.rand(10))))


class TestBatchSampler(unittest.TestCase):
    def test_batch_0(self):
        print(list(BatchSampler(SequentialSampler(np.random.rand(10)), batch_size=3, drop_last=False)))

    def test_batch_1(self):
        print(list(BatchSampler(SequentialSampler(np.random.rand(10)), batch_size=3, drop_last=True)))

    def test_batch_2(self):
        print(list(BatchSampler(RandomSampler(np.random.rand(10)), batch_size=3, drop_last=True)))
