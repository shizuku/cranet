import unittest

import numpy as np

from src import cranet


class TestNoGrad(unittest.TestCase):
    def test_1(self):
        b = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        self.assertTrue(b.requires_grad is True)
        with cranet.no_grad():
            a = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
            self.assertTrue(a.requires_grad is False)
        c = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        self.assertTrue(c.requires_grad is True)

    def test_2(self):
        a = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        b = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        e = a * b
        self.assertTrue(e.requires_grad is True)
        with cranet.no_grad():
            c = a + b
            self.assertTrue(c.requires_grad is False)
        d = a - b
        self.assertTrue(d.requires_grad is True)

    def test_3(self):
        a = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        b = cranet.tensor(np.random.rand(4, 5, 6), requires_grad=True)
        c, d = no_grad_test_3_func(a, b)
        e = a + b
        f = a - b
        self.assertTrue(c.requires_grad is False)
        self.assertTrue(d.requires_grad is False)
        self.assertTrue(e.requires_grad is True)
        self.assertTrue(f.requires_grad is True)


@cranet.no_grad()
def no_grad_test_3_func(a, b):
    return a + b, a - b
