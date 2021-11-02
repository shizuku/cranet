import os
import sys
import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.cranet.nn import functional as cranet_F

from src import cranet

from ..utils import teq


class TestMSELoss(unittest.TestCase):
    def test_mse_0(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.rand(64, 10)
            x_d = cranet.Tensor(x, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y, requires_grad=True)
            y_t = torch.tensor(y, requires_grad=True)
            l_d = cranet_F.mse_loss(x_d, y_d, reduction='mean')
            l_t = torch_F.mse_loss(x_t, y_t, reduction='mean')
            l_t.backward()
            l_d.backward()
            self.assertTrue(teq(l_d, l_t))
            self.assertTrue(teq(x_d.grad, x_t.grad))
            self.assertTrue(teq(y_d.grad, y_t.grad))

    def test_mse_1(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.rand(64, 10)
            x_d = cranet.Tensor(x, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y, requires_grad=True)
            y_t = torch.tensor(y, requires_grad=True)
            l_d = cranet_F.mse_loss(x_d, y_d, reduction='sum')
            l_t = torch_F.mse_loss(x_t, y_t, reduction='sum')
            l_t.backward()
            l_d.backward()
            self.assertTrue(teq(l_d, l_t, 2e-13))
            self.assertTrue(teq(x_d.grad, x_t.grad))
            self.assertTrue(teq(y_d.grad, y_t.grad))

    def test_mse_2(self):
        for _ in range(100):
            a = np.random.rand(64, 10)
            b = np.random.rand(64, 10)
            x_d = cranet.Tensor(a, requires_grad=True)
            x_t = torch.tensor(a, requires_grad=True)
            y_d = cranet.Tensor(b, requires_grad=True)
            y_t = torch.tensor(b, requires_grad=True)
            l_d = cranet_F.mse_loss(x_d, y_d, reduction='none')
            l_t = torch_F.mse_loss(x_t, y_t, reduction='none')
            g = np.random.rand(64, 10)
            l_t.backward(torch.tensor(g))
            l_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(y_d, y_t))
            self.assertTrue(teq(x_d.grad, x_t.grad))
            self.assertTrue(teq(y_d.grad, y_t.grad))


class TestBCELoss(unittest.TestCase):
    def test_binary_cross_entropy_0(self):
        for _ in range(1000):
            a = np.random.rand(10)
            b = np.random.rand(10)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = cranet_F.binary_cross_entropy(a0, b0, reduction='mean')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='mean')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad, 2e-11))

    def test_binary_cross_entropy_1(self):
        for _ in range(1000):
            a = np.random.rand(10)
            b = np.random.rand(10)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = cranet_F.binary_cross_entropy(a0, b0, reduction='sum')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='sum')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(teq(c0, c1, 2e-9))
            self.assertTrue(teq(a0.grad, a1.grad, 2e-10))

    def test_binary_cross_entropy_2(self):
        for _ in range(1000):
            a = np.random.rand(3, 5)
            b = np.random.rand(3, 5)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = cranet_F.binary_cross_entropy(a0, b0, reduction='mean')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='mean')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(teq(c0, c1, 2e-9))
            self.assertTrue(teq(a0.grad, a1.grad, 2e-9))

    def test_binary_cross_entropy_3(self):
        for _ in range(1000):
            a = np.random.rand(3, 5, 4)
            b = np.random.rand(3, 5, 4)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = cranet_F.binary_cross_entropy(a0, b0, reduction='sum')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='sum')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(teq(c0, c1, 2e-9))
            self.assertTrue(teq(a0.grad, a1.grad, 2e-9))


class TestCrossEntropyLoss(unittest.TestCase):
    def test_cross_entropy_0(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = cranet.Tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = cranet_F.cross_entropy(x_d, y_d, reduction='mean')
            l_t = torch_F.cross_entropy(x_t, y_t, reduction='mean')
            self.assertTrue(teq(l_d, l_t))
            l_t.backward()
            l_d.backward()
            self.assertTrue(teq(x_d.grad, x_t.grad))

    def test_cross_entropy_1(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = cranet.Tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = cranet_F.cross_entropy(x_d, y_d, reduction='sum')
            l_t = torch_F.cross_entropy(x_t, y_t, reduction='sum')
            self.assertTrue(teq(l_d, l_t, 2e-13))
            l_t.backward()
            l_d.backward()
            self.assertTrue(teq(x_d.grad, x_t.grad))

    def test_cross_entropy_2(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = cranet.Tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = cranet_F.cross_entropy(x_d, y_d, reduction='none')
            l_t = torch_F.cross_entropy(x_t, y_t, reduction='none')
            self.assertTrue(teq(l_d, l_t))
            g = np.random.rand(64)
            l_t.backward(torch.tensor(g))
            l_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(x_d.grad, x_t.grad))

    # def test_cross_entropy_3(self):
    #     for _ in range(100):
    #         x = np.random.rand(64, 10)
    #         y = np.random.randint(0, 10, [64])
    #         w = np.random.rand(64)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_d = cranet.Tensor(y)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         y_t = torch.tensor(y)
    #         l_d = cranet_F.cross_entropy(x_d, y_d, weight=cranet.Tensor(w))
    #         l_t = torch_F.cross_entropy(x_t, y_t, weight=torch.tensor(w))
    #         self.assertTrue(teq(l_d, l_t))
    #         l_t.backward()
    #         l_d.backward()
    #         self.assertTrue(teq(x_d.grad, x_t.grad))


class TestNLLLoss(unittest.TestCase):
    def test_nll_loss_0(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_c = cranet.tensor(x, requires_grad=True)
            y_c = cranet.tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_c = cranet_F.nll_loss(x_c, y_c, reduction='mean')
            l_t = torch_F.nll_loss(x_t, y_t, reduction='mean')
            self.assertTrue(teq(l_c, l_t))
            l_t.backward()
            l_c.backward()
            self.assertTrue(teq(x_c.grad, x_t.grad))

    def test_nll_loss_1(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = cranet.Tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = cranet_F.nll_loss(x_d, y_d, reduction='sum')
            l_t = torch_F.nll_loss(x_t, y_t, reduction='sum')
            self.assertTrue(teq(l_d, l_t, 2e-14))
            l_t.backward()
            l_d.backward()
            self.assertTrue(teq(x_d.grad, x_t.grad))

    def test_nll_loss_2(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = cranet.Tensor(x, requires_grad=True)
            y_d = cranet.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = cranet_F.nll_loss(x_d, y_d, reduction='none')
            l_t = torch_F.nll_loss(x_t, y_t, reduction='none')
            self.assertTrue(teq(l_d, l_t))
            g = np.random.rand(64)
            l_t.backward(torch.tensor(g))
            l_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(x_d.grad, x_t.grad))

    # def test_nll_loss_3(self):
    #     for _ in range(100):
    #         x = np.random.rand(64, 10)
    #         y = np.random.randint(0, 10, [64])
    #         w = np.random.rand(64)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_d = cranet.Tensor(y)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         y_t = torch.tensor(y)
    #         l_d = cranet_F.nll_loss(x_d, y_d, weight=cranet.Tensor(w))
    #         l_t = torch_F.nll_loss(x_t, y_t, weight=torch.tensor(w))
    #         self.assertTrue(teq(l_d, l_t))
    #         l_t.backward()
    #         l_d.backward()
    #         self.assertTrue(teq(x_d.grad, x_t.grad))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
