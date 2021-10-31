import os
import random
import sys
import unittest

import numpy as np

import torch
from torch.nn import functional as torch_F

from src import cranet

from ..utils import np_feq


class TestTensorSum(unittest.TestCase):
    def test_sum_0(self):
        for _ in range(100):
            a = np.random.rand(100)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.sum()
            b1 = a1.sum()
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13))
            b0.backward(cranet.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_sum_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.sum()
            b1 = a1.sum()
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"{b0.numpy()}\n{b1.detach().numpy()}")
            # print(np.random.rand().shape)
            delta = np.random.randn()
            delta0 = cranet.Tensor(delta, requires_grad=True)
            delta1 = torch.tensor(delta, requires_grad=True)
            b0.backward(delta0)
            b1.backward(delta1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-13))

    def test_sum_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(len(a.shape))
            b0 = a0.sum(axis)
            b1 = a1.sum(axis)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"cranet:\n{b0.numpy().shape}\ntorch:{b1.detach().numpy().shape}")
            delta = np.random.randn(*b0.shape)
            delta0 = cranet.Tensor(delta, requires_grad=True)
            delta1 = torch.tensor(delta, requires_grad=True)
            b0.backward(delta0)
            b1.backward(delta1)

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_sum_3(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            axis = (0, 1, 3)
            b0 = a0.sum(axis)
            b1 = a1.sum(axis)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"cranet:\n{b0.numpy().shape}\ntorch:{b1.detach().numpy().shape}")
            b1.backward(torch.ones_like(b1))
            b0.backward(cranet.ones_like(b0))

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTensorMean(unittest.TestCase):
    def test_mean_0(self):
        for _ in range(100):
            a = np.random.rand(100)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.mean()
            b1 = a1.mean()
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"\ncranet:{b0.numpy()}\ntorch:{b1.detach().numpy()}")
            b0.backward(cranet.ones_like(b0))
            b1.backward(torch.ones_like(b1))

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_mean_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.mean()
            b1 = a1.mean()
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"{b0.numpy()}\n{b1.detach().numpy()}")
            b0.backward(cranet.ones_like(b0))
            b1.backward(torch.ones_like(b1))

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_mean_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 2, 1, 3)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(len(a.shape))
            b0 = a0.mean(axis)
            b1 = a1.mean(axis)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"\ncranet:\n{b0.numpy()}\ntorch:{b1.detach().numpy()}")
            delta = np.random.randn(*b0.shape)
            delta0 = cranet.Tensor(delta, requires_grad=True)
            delta1 = torch.tensor(delta, requires_grad=True)
            b0.backward(delta0)
            b1.backward(delta1)

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_mean_3(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            axis = (0, 1, 4)
            b0 = a0.mean(axis)
            b1 = a1.mean(axis)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"\ncranet:\n{b0.numpy()}\ntorch:{b1.detach().numpy()}")
            delta = np.random.randn(*b0.shape)
            delta0 = cranet.Tensor(delta, requires_grad=True)
            delta1 = torch.tensor(delta, requires_grad=True)
            b0.backward(delta0)
            b1.backward(delta1)

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTensorAdd(unittest.TestCase):
    def test_add_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 + b0
            c1 = a1 + b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

    def test_add_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 + b0
            c1 = a1 + b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))


class TestTensorSub(unittest.TestCase):
    def test_sub_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 - b0
            c1 = a1 - b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

    def test_sub_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 - b0
            c1 = a1 - b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())


class TestTensorNeg(unittest.TestCase):
    def test_neg(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)

            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = -a0
            b1 = -a1
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTensorMul(unittest.TestCase):
    def test_mul_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 * b0
            c1 = a1 * b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

    def test_mul_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 * b0
            c1 = a1 * b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

    def test_mul_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 * b0
            c1 = a1 * b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-10))


class TestTensorTruediv(unittest.TestCase):
    def test_truediv_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 / b0
            c1 = a1 / b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-10))

    def test_truediv_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 / b0
            c1 = a1 / b1
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-7),
                            f"{b0.grad.numpy() - b1.grad.detach().numpy()}{b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7}{np.argmin(b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7)}")


class TestTensorMatmul(unittest.TestCase):
    def test_matmul_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 10)
            b = np.random.rand(10, 8)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            x = a0 @ b0
            x.backward(cranet.ones_like(x))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            y = a1 @ b1
            y.backward(torch.ones_like(y))
            self.assertTrue(np_feq(x.numpy(), y.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.numpy()))

    def test_matmul_2_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 5)
            b = np.random.rand(3, 5, 8)
            g = np.random.rand(3, 8, 8)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))

    def test_matmul_2_2(self):
        for _ in range(1000):
            a = np.random.rand(3, 8, 5)
            b = np.random.rand(5, 8)
            g = np.random.rand(3, 8, 8)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))

    def test_matmul_3(self):
        for _ in range(1000):
            a = np.random.rand(3, 3, 3, 8, 5)
            b = np.random.rand(5, 8)
            g = np.random.rand(3, 3, 3, 8, 8)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))

    def test_matmul_4_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 5)
            b = np.random.rand(5)
            g = np.random.rand(8)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))

    def test_matmul_4_2(self):
        for _ in range(1000):
            a = np.random.rand(5)
            b = np.random.rand(5, 6)
            g = np.random.rand(6)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))

    def test_matmul_5_1(self):
        for _ in range(1000):
            a = np.random.rand(3, 4, 8, 5)
            b = np.random.rand(5)
            g = np.random.rand(3, 4, 8)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))

    def test_matmul_5_2(self):
        for _ in range(1000):
            a = np.random.rand(5)
            b = np.random.rand(3, 4, 5, 6)
            g = np.random.rand(3, 4, 6)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            g_d = cranet.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_t.backward(g_t)
            y_d.backward(g_d)
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.numpy(), 2e-13))
            self.assertTrue(np_feq(b_d.grad.numpy(), b_t.grad.numpy(), 2e-13))


class TestTensorPower(unittest.TestCase):
    def test_power_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = cranet.power(a0, b0)
            c1 = torch.pow(a1, b1)
            c1.backward(torch.ones_like(c1))
            c0.backward(cranet.ones_like(c0))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

    def test_power_1(self):
        for _ in range(10000):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = cranet.power(a0, b0)
            c1 = torch.pow(a1, b1)
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

    def test_power_2(self):
        for _ in range(10000):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            b = np.random.rand(1)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = cranet.power(a0, b0)
            c1 = torch.pow(a1, b1)
            c0.backward(cranet.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-11))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-11))


class TestTensorTranspose(unittest.TestCase):
    def test_transpose_0(self):
        for _ in range(100):
            a = np.random.rand(8)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.transpose(0, 0)
            b0.backward(cranet.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 0, 0)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_transpose_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.transpose(0, 1)
            b0.backward(cranet.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 0, 1)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_transpose_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.transpose(1, 3)
            b0.backward(cranet.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 1, 3)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_transpose_3(self):
        for _ in range(100):
            a = np.random.rand(7, 9, 2, 4, 8, 3)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.transpose(2, 0)
            b0.backward(cranet.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 2, 0)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())


class TestTensorPermute(unittest.TestCase):
    def test_permute_0(self):
        for _ in range(10):
            a = np.random.rand(16)
            axes = list(range(1))
            random.shuffle(axes)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.permute(axes)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.permute(axes)
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b0.backward(cranet.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_permute_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4)
            axes = list(range(2))
            random.shuffle(axes)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.permute(axes)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.permute(axes)
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b0.backward(cranet.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_permute_2(self):
        for _ in range(100):
            a = np.random.rand(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            axes = list(range(10))
            random.shuffle(axes)
            a0 = cranet.Tensor(a.copy(), requires_grad=True)
            b0 = a0.permute(axes)
            a1 = torch.tensor(a.copy(), requires_grad=True)
            b1 = a1.permute(axes)
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b1.backward(torch.ones_like(b1))
            b0.backward(cranet.ones_like(b0))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_permute_3(self):
        for _ in range(100):
            a = np.random.rand(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            axes = list(range(10))
            random.shuffle(axes)
            a0 = cranet.Tensor(a.copy(), requires_grad=True)
            b0 = a0.T
            a1 = torch.tensor(a.copy(), requires_grad=True)
            b1 = a1.T
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b1.backward(torch.ones_like(b1))
            b0.backward(cranet.ones_like(b0))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())


class TestTensorConcat(unittest.TestCase):
    def test_concat_0(self):
        for _ in range(1000):
            a1 = np.random.rand(1, 3, 4)
            a2 = np.random.rand(1, 3, 4)
            delta = np.random.rand(2, 3, 4)
            t1 = torch.tensor(a1.copy(), requires_grad=True)
            t2 = torch.tensor(a2.copy(), requires_grad=True)
            d1 = cranet.Tensor(a1.copy(), requires_grad=True)
            d2 = cranet.Tensor(a2.copy(), requires_grad=True)
            t0 = torch.cat((t1, t2), 0)
            d0 = cranet.concat((d1, d2), 0)
            self.assertTrue(np_feq(d0.numpy(), t0.detach().numpy()))
            t0.backward(torch.tensor(delta.copy(), requires_grad=True))
            d0.backward(cranet.Tensor(delta.copy(), requires_grad=True))
            self.assertTrue(np_feq(d2.grad.numpy(), t2.grad.detach().numpy()))
            self.assertTrue(np_feq(d1.grad.numpy(), t1.grad.detach().numpy()))

    def test_concat_1(self):
        for _ in range(1000):
            a1 = np.random.rand(1, 3, 4)
            a2 = np.random.rand(2, 3, 4)
            a3 = np.random.rand(3, 3, 4)
            a4 = np.random.rand(4, 3, 4)
            a5 = np.random.rand(2, 3, 4)
            delta = np.random.rand(12, 3, 4)
            t1 = torch.tensor(a1.copy(), requires_grad=True)
            t2 = torch.tensor(a2.copy(), requires_grad=True)
            t3 = torch.tensor(a3.copy(), requires_grad=True)
            t4 = torch.tensor(a4.copy(), requires_grad=True)
            t5 = torch.tensor(a5.copy(), requires_grad=True)
            d1 = cranet.Tensor(a1.copy(), requires_grad=True)
            d2 = cranet.Tensor(a2.copy(), requires_grad=True)
            d3 = cranet.Tensor(a3.copy(), requires_grad=True)
            d4 = cranet.Tensor(a4.copy(), requires_grad=True)
            d5 = cranet.Tensor(a5.copy(), requires_grad=True)
            t0 = torch.cat((t1, t2, t3, t4, t5), 0)
            d0 = cranet.concat((d1, d2, d3, d4, d5), 0)
            self.assertTrue(np_feq(d0.numpy(), t0.detach().numpy()))
            t0.backward(torch.tensor(delta.copy()))
            d0.backward(cranet.Tensor(delta.copy()))
            self.assertTrue(np_feq(d1.grad.numpy(), t1.grad.detach().numpy()))
            self.assertTrue(np_feq(d2.grad.numpy(), t2.grad.detach().numpy()))
            self.assertTrue(np_feq(d3.grad.numpy(), t3.grad.detach().numpy()))
            self.assertTrue(np_feq(d4.grad.numpy(), t4.grad.detach().numpy()))
            self.assertTrue(np_feq(d5.grad.numpy(), t5.grad.detach().numpy()))


class TestTensorReshape(unittest.TestCase):
    def test_reshape_0(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            z = np.random.rand(6, 2, 5)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.reshape(6, 2, 5)
            a1 = torch.tensor(a, requires_grad=True)
            # b1 = torch.reshape(a1.reshape(6, 2, 5), [6, 2, 5])
            b1 = a1.reshape(6, 2, 5)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_reshape_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
            z = np.random.rand(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = a0.reshape(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.reshape(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            # b1 = torch.reshape(a1, [3 * 4 * 5 * 6 * 7 * 8 * 9 * 10])
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTensorFlatten(unittest.TestCase):
    def test_flatten_0(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            z = np.random.rand(3 * 4 * 5)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.flatten(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.flatten(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_flatten_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
            z = np.random.rand(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.flatten(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.flatten(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTensorSlice(unittest.TestCase):
    def test_slice_0(self):
        for _ in range(100):
            a = np.random.rand(12, 12)
            z = np.random.rand(4, 4)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = a_d[2:6, 2:6]
            a_t = torch.tensor(a, requires_grad=True)
            b_t = a_t[2:6, 2:6]
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            b_t.backward(torch.tensor(z))
            b_d.backward(cranet.Tensor(z))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))


class TestTensorPad(unittest.TestCase):
    def test_pad_0(self):
        for _ in range(100):
            a = np.random.rand(5, 5)
            z = np.random.rand(11, 11)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.pad(a0, [[3, 3], [3, 3]])
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.pad(a1, (3, 3, 3, 3))
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_pad_1(self):
        for _ in range(100):
            a = np.random.rand(1, 3, 5, 5)
            z = np.random.rand(1, 3, 11, 11)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.pad(a0, [[0, 0], [0, 0], [3, 3], [3, 3]])
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.pad(a1, (3, 3, 3, 3, 0, 0, 0, 0,))
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(cranet.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestUnsqueeze(unittest.TestCase):
    def test_unsqueeze_0(self):
        ts = [cranet.zeros((3, 4)) for _ in range(100)]
        z = cranet.stack(ts)
        print(z.shape)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
