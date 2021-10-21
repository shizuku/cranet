import os
import random
import sys
import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.dpln.nn import functional as dpln_F

from src import dpln


def np_feq(a: np.ndarray, b: np.ndarray, epsilon: float = 2e-15) -> bool:
    return (np.abs(a - b) < epsilon).all()


class TestTensorL1(unittest.TestCase):
    def setUp(self):
        pass

    def test_sum(self):
        for _ in range(100):
            a = np.random.rand(100)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.sum()
            b1 = torch.sum(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13))
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.sum()
            b1 = a1.sum()
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"{b0.numpy()}\n{b1.detach().numpy()}")
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            axis = 1
            b0 = a0.sum(axis)
            b1 = a1.sum(axis)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy(), 2e-13), f"{b0.numpy()}\n{b1.detach().numpy()}")
            b1.backward(torch.ones_like(b1))
            b0.backward(dpln.ones_like(b0))

            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_add(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 + b0
            c1 = a1 + b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 + b0
            c1 = a1 + b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

    def test_sub(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 - b0
            c1 = a1 - b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 - b0
            c1 = a1 - b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

    def test_neg(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = -a0
            b1 = -a1
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_mul(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 * b0
            c1 = a1 * b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 * b0
            c1 = a1 * b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 * b0
            c1 = a1 * b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-10))

    def test_truediv(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 / b0
            c1 = a1 / b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-10))

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = a0 / b0
            c1 = a1 / b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-7),
                            f"{b0.grad.numpy() - b1.grad.detach().numpy()}{b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7}{np.argmin(b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7)}")


class TestTensorMatmul(unittest.TestCase):
    def test_matmul_0(self):
        for _ in range(1000):
            a = np.random.rand(8, 10)
            b = np.random.rand(10)
            g = np.random.rand(8)
            a_d = dpln.Tensor(a, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            g_d = dpln.Tensor(g)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            g_t = torch.tensor(g)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(np_feq(y_d.numpy(), y_t.detach().numpy()))
            y_d.backward(g_d)
            y_t.backward(g_t)
            self.assertTrue((a_d.grad.numpy() == a_t.grad.numpy()).all())
            self.assertTrue((b_d.grad.numpy() == b_t.grad.numpy()).all())

    def test_matmul_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 10)
            b = np.random.rand(10, 8)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            x = a0 @ b0
            x.backward(dpln.ones_like(x))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            y = a1 @ b1
            y.backward(torch.ones_like(y))
            self.assertTrue((x.numpy() == y.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.numpy()).all())

    def test_matmul_2_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 5)
            b = np.random.rand(3, 5, 8)
            g = np.random.rand(3, 8, 8)
            a_d = dpln.Tensor(a, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            g_d = dpln.Tensor(g)
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
            a_d = dpln.Tensor(a, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            g_d = dpln.Tensor(g)
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
            a_d = dpln.Tensor(a, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            g_d = dpln.Tensor(g)
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
    def test_power(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = dpln.power(a0, b0)
            c1 = torch.pow(a1, b1)
            c1.backward(torch.ones_like(c1))
            c0.backward(dpln.ones_like(c0))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

        for _ in range(10000):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = dpln.power(a0, b0)
            c1 = torch.pow(a1, b1)
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-10))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy()))

        for _ in range(10000):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            b = np.random.rand(1)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = dpln.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = dpln.power(a0, b0)
            c1 = torch.pow(a1, b1)
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy(), 2e-11))
            self.assertTrue(np_feq(b0.grad.numpy(), b1.grad.detach().numpy(), 2e-11))

    def test_transpose(self):
        for _ in range(100):
            a = np.random.rand(8)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.transpose(0, 0)
            b0.backward(dpln.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 0, 0)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.transpose(0, 1)
            b0.backward(dpln.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 0, 1)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.transpose(1, 3)
            b0.backward(dpln.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 1, 3)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(7, 9, 2, 4, 8, 3)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.transpose(2, 0)
            b0.backward(dpln.ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 2, 0)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_permute(self):
        for _ in range(10):
            a = np.random.rand(16)
            axes = list(range(1))
            random.shuffle(axes)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.permute(axes)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.permute(axes)
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4)
            axes = list(range(2))
            random.shuffle(axes)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.permute(axes)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.permute(axes)
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            axes = list(range(10))
            random.shuffle(axes)
            a0 = dpln.Tensor(a.copy(), requires_grad=True)
            b0 = a0.permute(axes)
            a1 = torch.tensor(a.copy(), requires_grad=True)
            b1 = a1.permute(axes)
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b1.backward(torch.ones_like(b1))
            b0.backward(dpln.ones_like(b0))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            axes = list(range(10))
            random.shuffle(axes)
            a0 = dpln.Tensor(a.copy(), requires_grad=True)
            b0 = a0.T
            a1 = torch.tensor(a.copy(), requires_grad=True)
            b1 = a1.T
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b1.backward(torch.ones_like(b1))
            b0.backward(dpln.ones_like(b0))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_concat(self):
        for _ in range(1000):
            a1 = np.random.rand(1, 3, 4)
            a2 = np.random.rand(1, 3, 4)
            delta = np.random.rand(2, 3, 4)
            t1 = torch.tensor(a1.copy(), requires_grad=True)
            t2 = torch.tensor(a2.copy(), requires_grad=True)
            d1 = dpln.Tensor(a1.copy(), requires_grad=True)
            d2 = dpln.Tensor(a2.copy(), requires_grad=True)
            t0 = torch.cat((t1, t2), 0)
            d0 = dpln.concat((d1, d2), 0)
            self.assertTrue(np_feq(d0.numpy(), t0.detach().numpy()))
            t0.backward(torch.tensor(delta.copy(), requires_grad=True))
            d0.backward(dpln.Tensor(delta.copy(), requires_grad=True))
            self.assertTrue(np_feq(d2.grad.numpy(), t2.grad.detach().numpy()))
            self.assertTrue(np_feq(d1.grad.numpy(), t1.grad.detach().numpy()))

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
            d1 = dpln.Tensor(a1.copy(), requires_grad=True)
            d2 = dpln.Tensor(a2.copy(), requires_grad=True)
            d3 = dpln.Tensor(a3.copy(), requires_grad=True)
            d4 = dpln.Tensor(a4.copy(), requires_grad=True)
            d5 = dpln.Tensor(a5.copy(), requires_grad=True)
            t0 = torch.cat((t1, t2, t3, t4, t5), 0)
            d0 = dpln.concat((d1, d2, d3, d4, d5), 0)
            self.assertTrue(np_feq(d0.numpy(), t0.detach().numpy()))
            t0.backward(torch.tensor(delta.copy()))
            d0.backward(dpln.Tensor(delta.copy()))
            self.assertTrue(np_feq(d1.grad.numpy(), t1.grad.detach().numpy()))
            self.assertTrue(np_feq(d2.grad.numpy(), t2.grad.detach().numpy()))
            self.assertTrue(np_feq(d3.grad.numpy(), t3.grad.detach().numpy()))
            self.assertTrue(np_feq(d4.grad.numpy(), t4.grad.detach().numpy()))
            self.assertTrue(np_feq(d5.grad.numpy(), t5.grad.detach().numpy()))

    def test_reshape(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            z = np.random.rand(6, 2, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.reshape(6, 2, 5)
            a1 = torch.tensor(a, requires_grad=True)
            # b1 = torch.reshape(a1.reshape(6, 2, 5), [6, 2, 5])
            b1 = a1.reshape(6, 2, 5)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
            z = np.random.rand(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = a0.reshape(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.reshape(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            # b1 = torch.reshape(a1, [3 * 4 * 5 * 6 * 7 * 8 * 9 * 10])
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_flatten(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            z = np.random.rand(3 * 4 * 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.flatten(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.flatten(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
            z = np.random.rand(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.flatten(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.flatten(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTensorSlice(unittest.TestCase):
    def test_slice_0(self):
        for _ in range(100):
            a = np.random.rand(12, 12)
            z = np.random.rand(4, 4)
            a_d = dpln.Tensor(a, requires_grad=True)
            b_d = a_d[2:6, 2:6]
            a_t = torch.tensor(a, requires_grad=True)
            b_t = a_t[2:6, 2:6]
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            b_t.backward(torch.tensor(z))
            b_d.backward(dpln.Tensor(z))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))


class TestTensorPad(unittest.TestCase):
    def test_pad_0(self):
        for _ in range(100):
            a = np.random.rand(5, 5)
            z = np.random.rand(11, 11)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.pad(a0, [[3, 3], [3, 3]])
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.pad(a1, (3, 3, 3, 3))
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_pad_1(self):
        for _ in range(100):
            a = np.random.rand(1, 3, 5, 5)
            z = np.random.rand(1, 3, 11, 11)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.pad(a0, [[0, 0], [0, 0], [3, 3], [3, 3]])
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.pad(a1, (3, 3, 3, 3, 0, 0, 0, 0,))
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.Tensor(z))
            b1.backward(torch.tensor(z))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestFunction(unittest.TestCase):
    def test_log(self):
        for _ in range(100):
            a = np.random.rand(3)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.log(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.log(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()), f"{b0},{b1}")
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.log(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.log(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()), f"{b0},{b1}")
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_exp(self):
        for _ in range(100):
            a = np.random.rand(8)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.exp(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.exp(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.exp(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.exp(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_relu(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln_F.relu(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.relu(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_sigmoid(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln_F.sigmoid(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.sigmoid(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_softmax(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln_F.softmax(a0, axis=-1)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.softmax(a1, dim=-1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_tanh(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln_F.tanh(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.tanh(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_flatten(self):
        for _ in range(1):
            a = np.random.rand(2, 2, 2)
            t = torch.tensor(a, requires_grad=True)
            z = torch.flatten(t)
            z.backward(torch.ones_like(z))
            print()
            print(t.grad)
            print(t.grad.size())
            print(z.size())


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
