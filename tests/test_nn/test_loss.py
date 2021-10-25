import os
import sys
import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.dpln.nn import functional as dpln_F

from src import dpln

from ..utils import np_feq


class TestMse(unittest.TestCase):
    def test_mse_0(self):
        for _ in range(100):
            a = np.random.rand(10)
            b = np.random.rand(10)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = dpln_F.mse_loss(a0, b0, reduction='mean')
            c1 = torch_F.mse_loss(a1, b1, reduction='mean')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

    def test_mse_1(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            b = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = dpln_F.mse_loss(a0, b0, reduction='mean')
            c1 = torch_F.mse_loss(a1, b1, reduction='mean')
            c1.backward(torch.ones_like(c1))
            c0.backward(dpln.ones_like(c0))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))


class TestBCELoss(unittest.TestCase):
    def test_binary_cross_entropy_0(self):
        for _ in range(1000):
            a = np.random.rand(10)
            b = np.random.rand(10)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = dpln_F.binary_cross_entropy(a0, b0, reduction='mean')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='mean')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-11))

    def test_binary_cross_entropy_1(self):
        for _ in range(1000):
            a = np.random.rand(10)
            b = np.random.rand(10)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = dpln_F.binary_cross_entropy(a0, b0, reduction='sum')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='sum')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy(), 2e-9))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))

    def test_binary_cross_entropy_2(self):
        for _ in range(1000):
            a = np.random.rand(3, 5)
            b = np.random.rand(3, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = dpln_F.binary_cross_entropy(a0, b0, reduction='mean')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='mean')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy(), 2e-9))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-9))

    def test_binary_cross_entropy_3(self):
        for _ in range(1000):
            a = np.random.rand(3, 5, 4)
            b = np.random.rand(3, 5, 4)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b)
            c0 = dpln_F.binary_cross_entropy(a0, b0, reduction='sum')
            c1 = torch_F.binary_cross_entropy(a1, b1, reduction='sum')
            c0.zero_grad()
            c1.backward()
            c0.backward()
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy(), 2e-9))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-9))


class TestCrossEntropy(unittest.TestCase):
    def test_cross_entropy_0(self):
        pass
        # y = np.array([1, 0, 0])
        # y1 = np.array([0.7, 0.2, 0.1])
        # Y = dpln.Tensor(y)
        # Y1 = dpln.Tensor(y1)
        # l = dpln_F.cross_entropy(Y, Y1, reduction="sum")
        # print(l)
        # # print(f"loss: {l:.4f}")


class TestNllLoss(unittest.TestCase):
    def test_nll_loss_0(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = dpln.Tensor(x, requires_grad=True)
            y_d = dpln.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = dpln_F.nll_loss(x_d, y_d, reduction='mean')
            l_t = torch_F.nll_loss(x_t, y_t, reduction='mean')
            self.assertTrue(np_feq(l_d.detach().numpy(), l_t.detach().numpy()))
            l_t.backward()
            l_d.backward()
            self.assertTrue(np_feq(x_d.grad.detach().numpy(), x_t.grad.detach().numpy()))

    def test_nll_loss_1(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = dpln.Tensor(x, requires_grad=True)
            y_d = dpln.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = dpln_F.nll_loss(x_d, y_d, reduction='sum')
            l_t = torch_F.nll_loss(x_t, y_t, reduction='sum')
            self.assertTrue(np_feq(l_d.detach().numpy(), l_t.detach().numpy(), 2e-14))
            l_t.backward()
            l_d.backward()
            self.assertTrue(np_feq(x_d.grad.detach().numpy(), x_t.grad.detach().numpy()))

    def test_nll_loss_2(self):
        for _ in range(100):
            x = np.random.rand(64, 10)
            y = np.random.randint(0, 10, [64])
            x_d = dpln.Tensor(x, requires_grad=True)
            y_d = dpln.Tensor(y)
            x_t = torch.tensor(x, requires_grad=True)
            y_t = torch.tensor(y)
            l_d = dpln_F.nll_loss(x_d, y_d, reduction='none')
            l_t = torch_F.nll_loss(x_t, y_t, reduction='none')
            self.assertTrue(np_feq(l_d.detach().numpy(), l_t.detach().numpy()))
            g = np.random.rand(64)
            l_t.backward(torch.tensor(g))
            l_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(x_d.grad.detach().numpy(), x_t.grad.detach().numpy()))

    # def test_nll_loss_3(self):
    #     for _ in range(100):
    #         x = np.random.rand(64, 10)
    #         y = np.random.randint(0, 10, [64])
    #         w = np.random.rand(64)
    #         x_d = dpln.Tensor(x, requires_grad=True)
    #         y_d = dpln.Tensor(y)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         y_t = torch.tensor(y)
    #         l_d = dpln_F.nll_loss(x_d, y_d, weight=dpln.Tensor(w))
    #         l_t = torch_F.nll_loss(x_t, y_t, weight=torch.tensor(w))
    #         self.assertTrue(np_feq(l_d.detach().numpy(), l_t.detach().numpy()))
    #         l_t.backward()
    #         l_d.backward()
    #         self.assertTrue(np_feq(x_d.grad.detach().numpy(), x_t.grad.detach().numpy()))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
