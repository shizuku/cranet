import os
import sys
import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.cranet.nn import functional as cranet_F

from src import cranet

from ..utils import np_feq


class TestDropout(unittest.TestCase):
    def test_dropout_0(self):
        a = np.random.rand(3, 3)
        a_d = cranet.Tensor(a, requires_grad=True)
        b_d = cranet_F.dropout(a_d)
        print(b_d)
        self.assertTrue(b_d.shape == (3, 3))
        g = np.random.rand(3, 3)
        g_d = cranet.Tensor(g)
        b_d.backward(g_d)
        print(a_d.grad)


class TestMaxPool2d(unittest.TestCase):
    def test_max_pool_0(self):
        a = np.random.rand(5, 3, 8, 8)
        a_d = cranet.Tensor(a, requires_grad=True)
        a_t = torch.tensor(a, requires_grad=True)
        b_t = torch_F.max_pool2d(a_t, 2, 2)
        b_d = cranet_F.max_pool2d(a_d, 2, 2)
        self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
        g = np.random.rand(5, 3, 4, 4)
        b_t.backward(torch.tensor(g))
        b_d.backward(cranet.Tensor(g))
        self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))

    def test_max_pool_1(self):
        a = np.random.rand(5, 3, 32, 32)
        a_d = cranet.Tensor(a, requires_grad=True)
        a_t = torch.tensor(a, requires_grad=True)
        b_t = torch_F.max_pool2d(a_t, 2, 2)
        b_d = cranet_F.max_pool2d(a_d, 2, 2)
        self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
        g = np.random.rand(5, 3, 16, 16)
        b_t.backward(torch.tensor(g))
        b_d.backward(cranet.Tensor(g))
        self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))


class TestFlatten(unittest.TestCase):
    def test_0(self):
        a = cranet.zeros((3, 4, 5, 6, 7, 8, 9))
        b = cranet_F.flatten(a)
        self.assertTrue(b.shape == (3, 4 * 5 * 6 * 7 * 8 * 9))

    def test_1(self):
        a = cranet.zeros((1, 2, 3, 4, 5, 6, 7, 8, 9))
        b = cranet_F.flatten(a, 2, -2)
        self.assertTrue(b.shape == (1, 2, 3 * 4 * 5 * 6 * 7 * 8, 9))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
