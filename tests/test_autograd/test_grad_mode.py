import unittest

import numpy as np

from src import cranet


class TestNoGrad(unittest.TestCase):
    def test_1(self):
        print(cranet.is_grad_enabled())
        with cranet.no_grad():
            print(cranet.is_grad_enabled())
            a = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
            self.assertTrue(a.requires_grad is False)

    def test_0(self):
        a = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        b = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        with cranet.no_grad():
            c = a + b
            self.assertTrue(c.requires_grad is False)
