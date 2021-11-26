import os
import random
import sys
import unittest

import numpy as np

import torch
from torch.nn import functional as torch_F
from cranet.nn import functional as cranet_F

from src import cranet

from ..utils import teq


class TestTensorTemplate(unittest.TestCase):
    def test_autograd_template_0(self):
        a = cranet.random.normal((2, 3), requires_grad=True)
        result = a.sum()
        result.backward()
        print(a)
        print()
        print(a.grad)

    def test_autograd_template_1(self):
        x = cranet.ones(5)
        y = cranet.zeros(3)
        w = cranet.random.normal((5, 3), requires_grad=True)
        b = cranet.random.normal(3, requires_grad=True)
        z = x @ w + b

        loss = cranet.nn.functional.mse_loss(z, y)
        loss.backward()

        print()
        print(f"w: {w.grad} \nb: {b.grad}")


class TestTensorSum(unittest.TestCase):
    def test_sum_0(self):
        for _ in range(100):
            a = np.random.rand(100)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = a_c.sum()
            b_t = a_t.sum()
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand()
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-5))

    def test_sum_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = a_c.sum()
            b_t = a_t.sum()
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.randn()
            g_c = cranet.tensor(g)
            g_t = torch.tensor(g)
            b_c.backward(g_c)
            b_t.backward(g_t)
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-6))

    def test_sum_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(len(a.shape))
            b_c = a_c.sum(axis)
            b_t = a_t.sum(axis)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.randn(*b_c.shape)
            g_c = cranet.tensor(g)
            g_t = torch.tensor(g)
            b_c.backward(g_c)
            b_t.backward(g_t)
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-6))

    def test_sum_3(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = (0, 1, 3)
            b_c = a_c.sum(axis)
            b_t = a_t.sum(axis)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand(5, 7, 8)
            b_t.backward(torch.tensor(g))
            b_c.backward(cranet.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad))

    def test_sum_4(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = (0, 1, 3)
            b_c = cranet.sum(a_c, dim=axis, keepdim=True)
            b_t = torch.sum(a_t, dim=axis, keepdim=True)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand(1, 1, 5, 1, 7, 8)
            b_t.backward(torch.tensor(g))
            b_c.backward(cranet.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad))


class TestTensorMean(unittest.TestCase):
    def test_mean_0(self):
        for _ in range(100):
            a = np.random.rand(100)
            a0 = cranet.tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.mean()
            b1 = a1.mean()
            self.assertTrue(teq(b0, b1, 1e-13))
            g = np.random.rand()
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad, 1e-5))

    def test_mean_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = cranet.tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.mean()
            b1 = a1.mean()
            self.assertTrue(teq(b0, b1, 1e-13))
            g = np.random.rand()
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad, 1e-5))

    def test_mean_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 2, 1, 3)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(len(a.shape))
            b_c = a_c.mean(axis)
            b_t = a_t.mean(axis)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.randn(*b_c.shape)
            g_c = cranet.tensor(g)
            g_t = torch.tensor(g)
            b_c.backward(g_c)
            b_t.backward(g_t)
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-5))

    def test_mean_3(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = (0, 1, 4)
            b_c = a_c.mean(axis)
            b_t = a_t.mean(axis)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.randn(*b_c.shape)
            g_c = cranet.tensor(g)
            g_t = torch.tensor(g)
            b_c.backward(g_c)
            b_t.backward(g_t)
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-5))

    def test_mean_4(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = (0, 1, 4)
            b_c = a_c.mean(axis, keepdim=True)
            b_t = a_t.mean(axis, keepdim=True)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.randn(*b_c.shape)
            g_c = cranet.tensor(g)
            g_t = torch.tensor(g)
            b_c.backward(g_c)
            b_t.backward(g_t)
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-5))


class TestTensorVar(unittest.TestCase):
    def test_var_0(self):
        for _ in range(100):
            a = np.random.rand(3, 4)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = cranet.var(a_c, unbiased=False)
            b_t = torch.var(a_t, unbiased=False)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand()
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-6))

    def test_var_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = cranet.var(a_c, dim=2, unbiased=False)
            b_t = torch.var(a_t, dim=2, unbiased=False)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand(3, 4, 6)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-6))

    def test_var_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            dim = (2, 3, 1)
            b_c = cranet.var(a_c, dim=dim, unbiased=False)
            b_t = torch.var(a_t, dim=dim, unbiased=False)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand(*b_c.shape)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-6))

    def test_var_3(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            dim = (2, 3, 1)
            b_c = cranet.var(a_c, dim=dim, keepdim=True, unbiased=False)
            b_t = torch.var(a_t, dim=dim, keepdim=True, unbiased=False)
            self.assertTrue(teq(b_c, b_t, 1e-13))
            g = np.random.rand(*b_c.shape)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-6))


class TestTensorAdd(unittest.TestCase):
    def test_add_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a_c = cranet.tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = cranet.tensor(b, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            c_c = a_c + b_c
            c_t = a_t + b_t
            g = np.random.rand(1)
            c_c.backward(cranet.tensor(g))
            c_t.backward(torch.tensor(g))
            self.assertTrue(teq(c_c, c_t))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-5))
            self.assertTrue(teq(b_c.grad, b_t.grad, 1e-5))

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
            g = np.random.rand(3, 4, 5, 6)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad))


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
            g = np.random.rand(1)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad))

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
            g = np.random.rand(3, 4, 5, 6)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad))


class TestTensorNeg(unittest.TestCase):
    def test_neg(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = -a0
            b1 = -a1
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(3, 4, 5)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


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
            g = np.random.rand(1)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad))

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
            g = np.random.rand(3, 4, 5, 6)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad))

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
            g = np.random.rand(3, 4, 5, 6)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad, 1e-10))


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
            g = np.random.rand(1)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad, 1e-10))
            self.assertTrue(teq(b0.grad, b1.grad, 1e-10))

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
            g = np.random.rand(3, 4, 5, 6)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad, 1e-10))
            self.assertTrue(teq(b0.grad, b1.grad, 1e-7))


class TestTensorMatmul(unittest.TestCase):
    def test_matmul_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 10)
            b = np.random.rand(10, 8)
            a_c = cranet.tensor(a, requires_grad=True)
            b_c = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_c = a_c @ b_c
            y_t = a_t @ b_t
            g = np.random.rand(8, 8)
            y_c.backward(cranet.tensor(g))
            y_t.backward(torch.tensor(g))
            self.assertTrue(teq(y_c, y_t, 1e-13))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-10))
            self.assertTrue(teq(b_c.grad, b_t.grad, 1e-10))

    def test_matmul_2_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 5)
            b = np.random.rand(3, 5, 8)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t))
            g = np.random.rand(3, 8, 8)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))

    def test_matmul_2_2(self):
        for _ in range(1000):
            a = np.random.rand(3, 8, 5)
            b = np.random.rand(5, 8)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t))
            g = np.random.rand(3, 8, 8)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))

    def test_matmul_3(self):
        for _ in range(1000):
            a = np.random.rand(3, 3, 3, 8, 5)
            b = np.random.rand(5, 8)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t, 1e-13))
            g = np.random.rand(3, 3, 3, 8, 8)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))

    def test_matmul_4_1(self):
        for _ in range(1000):
            a = np.random.rand(8, 5)
            b = np.random.rand(5)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t))
            g = np.random.rand(8)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))

    def test_matmul_4_2(self):
        for _ in range(1000):
            a = np.random.rand(5)
            b = np.random.rand(5, 6)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t))
            g = np.random.rand(6)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))

    def test_matmul_5_1(self):
        for _ in range(1000):
            a = np.random.rand(3, 4, 8, 5)
            b = np.random.rand(5)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = cranet.tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t))
            g = np.random.rand(3, 4, 8)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))

    def test_matmul_5_2(self):
        for _ in range(1000):
            a = np.random.rand(5)
            b = np.random.rand(3, 4, 5, 6)
            a_d = cranet.Tensor(a, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            y_d = a_d @ b_d
            y_t = a_t @ b_t
            self.assertTrue(teq(y_d, y_t))
            g = np.random.rand(3, 4, 6)
            y_t.backward(torch.tensor(g))
            y_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 1e-13))
            self.assertTrue(teq(b_d.grad, b_t.grad, 1e-13))


class TestTensorPower(unittest.TestCase):
    def test_power_0(self):
        for _ in range(100):
            a = np.random.rand(1)
            b = np.random.rand(1)
            a0 = cranet.tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = cranet.power(a0, b0)
            c1 = torch.pow(a1, b1)
            self.assertTrue(teq(c0, c1))
            g = np.random.rand(1)
            c1.backward(torch.tensor(g))
            c0.backward(cranet.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))
            self.assertTrue(teq(b0.grad, b1.grad))

    def test_power_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            b = np.random.rand(3, 4, 5, 6)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.Tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = cranet.power(a0, b0)
            c1 = torch.pow(a1, b1)
            self.assertTrue(teq(c0, c1))
            g = np.random.rand(3, 4, 5, 6)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad, 1e-10))
            self.assertTrue(teq(b0.grad, b1.grad, 1e-10))

    def test_power_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8)
            b = np.random.rand(1)
            a0 = cranet.tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = cranet.tensor(b, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            c0 = cranet.power(a0, b0)
            c1 = torch.pow(a1, b1)
            g = np.random.rand(3, 4, 5, 6, 7, 8)
            c0.backward(cranet.tensor(g))
            c1.backward(torch.tensor(g))
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad, 1e-10))
            self.assertTrue(teq(b0.grad, b1.grad, 1e-10))


class TestTensorTranspose(unittest.TestCase):
    def test_transpose_0(self):
        for _ in range(100):
            a = np.random.rand(8)
            a_t = torch.tensor(a, requires_grad=True)
            a_c = cranet.tensor(a, requires_grad=True)
            b_t = torch.transpose(a_t, 0, 0)
            b_c = cranet.transpose(a_c, 0, 0)
            self.assertTrue(teq(b_c, b_t, 1e-15))
            g = np.random.rand(8)
            b_t.backward(torch.tensor(g))
            b_c.backward(cranet.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))

    def test_transpose_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4)
            a_t = torch.tensor(a, requires_grad=True)
            a_c = cranet.Tensor(a, requires_grad=True)
            b_t = torch.transpose(a_t, 0, 1)
            b_c = cranet.transpose(a_c, 0, 1)
            self.assertTrue(teq(b_c, b_t, 1e-15))
            g = np.random.rand(4, 3)
            b_t.backward(torch.tensor(g))
            b_c.backward(cranet.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))

    def test_transpose_2(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a_t = torch.tensor(a, requires_grad=True)
            a_c = cranet.Tensor(a, requires_grad=True)
            b_t = torch.transpose(a_t, 0, 1)
            b_c = cranet.transpose(a_c, 0, 1)
            self.assertTrue(teq(b_c, b_t, 1e-15))
            g = np.random.rand(4, 3, 5, 6)
            b_t.backward(torch.tensor(g))
            b_c.backward(cranet.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))

    def test_transpose_3(self):
        for _ in range(100):
            a = np.random.rand(7, 9, 2, 4, 8, 3)
            a_t = torch.tensor(a, requires_grad=True)
            a_c = cranet.tensor(a, requires_grad=True)
            b_t = torch.transpose(a_t, 0, 1)
            b_c = cranet.transpose(a_c, 0, 1)
            self.assertTrue(teq(b_c, b_t, 1e-15))
            g = np.random.rand(*b_c.shape)
            b_t.backward(torch.tensor(g))
            b_c.backward(cranet.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))


class TestTensorPermute(unittest.TestCase):
    def test_permute_0(self):
        for _ in range(10):
            a = np.random.rand(16)
            axes = list(range(1))
            random.shuffle(axes)
            a_c = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = a_c.permute(axes)
            b_t = a_t.permute(axes)
            self.assertTrue(teq(b_c, b_t))
            g = np.random.rand(*b_c.shape)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))

    def test_permute_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4)
            axes = list(range(2))
            random.shuffle(axes)
            a_c = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = a_c.permute(axes)
            b_t = a_t.permute(axes)
            self.assertTrue(teq(b_c, b_t))
            g = np.random.rand(*b_c.shape)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))

    def test_permute_2(self):
        for _ in range(100):
            a = np.random.rand(2, 3, 4, 5, 6)
            axes = list(range(5))
            random.shuffle(axes)
            a_c = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = a_c.permute(axes)
            b_t = a_t.permute(axes)
            self.assertTrue(teq(b_c, b_t))
            g = np.random.rand(*b_c.shape)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))

    def test_permute_3(self):
        for _ in range(100):
            a = np.random.rand(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            axes = list(range(10))
            random.shuffle(axes)
            a_c = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_c = a_c.permute(axes)
            b_t = a_t.permute(axes)
            self.assertTrue(teq(b_c, b_t))
            g = np.random.rand(*b_c.shape)
            b_c.backward(cranet.tensor(g))
            b_t.backward(torch.tensor(g))
            self.assertTrue(teq(a_c.grad, a_t.grad, 1e-15))


class TestTensorConcat(unittest.TestCase):
    def test_concat_0(self):
        for _ in range(1000):
            a1 = np.random.rand(1, 3, 4)
            a2 = np.random.rand(1, 3, 4)
            t1 = torch.tensor(a1, requires_grad=True)
            t2 = torch.tensor(a2, requires_grad=True)
            d1 = cranet.tensor(a1, requires_grad=True)
            d2 = cranet.tensor(a2, requires_grad=True)
            t0 = torch.cat((t1, t2), 0)
            d0 = cranet.concat((d1, d2), 0)
            self.assertTrue(teq(d0, t0))
            g = np.random.rand(2, 3, 4)
            t0.backward(torch.tensor(g, requires_grad=True))
            d0.backward(cranet.tensor(g, requires_grad=True))
            self.assertTrue(teq(d2.grad, t2.grad))
            self.assertTrue(teq(d1.grad, t1.grad))

    def test_concat_1(self):
        for _ in range(1000):
            a1 = np.random.rand(1, 3, 4)
            a2 = np.random.rand(2, 3, 4)
            a3 = np.random.rand(3, 3, 4)
            a4 = np.random.rand(4, 3, 4)
            a5 = np.random.rand(2, 3, 4)
            t1 = torch.tensor(a1.copy(), requires_grad=True)
            t2 = torch.tensor(a2.copy(), requires_grad=True)
            t3 = torch.tensor(a3.copy(), requires_grad=True)
            t4 = torch.tensor(a4.copy(), requires_grad=True)
            t5 = torch.tensor(a5.copy(), requires_grad=True)
            d1 = cranet.tensor(a1.copy(), requires_grad=True)
            d2 = cranet.tensor(a2.copy(), requires_grad=True)
            d3 = cranet.tensor(a3.copy(), requires_grad=True)
            d4 = cranet.tensor(a4.copy(), requires_grad=True)
            d5 = cranet.tensor(a5.copy(), requires_grad=True)
            t0 = torch.cat((t1, t2, t3, t4, t5), 0)
            d0 = cranet.concat((d1, d2, d3, d4, d5), 0)
            self.assertTrue(teq(d0, t0))
            g = np.random.rand(12, 3, 4)
            t0.backward(torch.tensor(g))
            d0.backward(cranet.tensor(g))
            self.assertTrue(teq(d1.grad, t1.grad))
            self.assertTrue(teq(d2.grad, t2.grad))
            self.assertTrue(teq(d3.grad, t3.grad))
            self.assertTrue(teq(d4.grad, t4.grad))
            self.assertTrue(teq(d5.grad, t5.grad))


class TestTensorReshape(unittest.TestCase):
    def test_reshape_0(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = cranet.tensor(a, requires_grad=True)
            b0 = a0.reshape(6, 2, 5)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.reshape(6, 2, 5)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(6, 2, 5)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))

    def test_reshape_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
            a0 = cranet.tensor(a, requires_grad=True)
            b0 = a0.reshape(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = a1.reshape(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestTensorFlatten(unittest.TestCase):
    def test_flatten_0(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = cranet.tensor(a, requires_grad=True)
            b0 = cranet.flatten(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.flatten(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(3 * 4 * 5)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))

    def test_flatten_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
            a0 = cranet.tensor(a, requires_grad=True)
            b0 = cranet.flatten(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.flatten(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(3 * 4 * 5 * 6 * 7 * 8 * 9 * 10)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestTensorSlice(unittest.TestCase):
    def test_slice_0(self):
        for _ in range(100):
            a = np.random.rand(12, 12)
            a_d = cranet.tensor(a, requires_grad=True)
            b_d = a_d[2:6, 2:6]
            a_t = torch.tensor(a, requires_grad=True)
            b_t = a_t[2:6, 2:6]
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand(4, 4)
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad))


class TestTensorPad(unittest.TestCase):
    def test_pad_0(self):
        for _ in range(100):
            a = np.random.rand(5, 5)
            a0 = cranet.tensor(a, requires_grad=True)
            b0 = cranet.pad(a0, [[3, 3], [3, 3]])
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.pad(a1, (3, 3, 3, 3))
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(11, 11)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))

    def test_pad_1(self):
        for _ in range(100):
            a = np.random.rand(1, 3, 5, 5)
            a0 = cranet.tensor(a, requires_grad=True)
            b0 = cranet.pad(a0, [[0, 0], [0, 0], [3, 3], [3, 3]])
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch_F.pad(a1, (3, 3, 3, 3, 0, 0, 0, 0,))
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(1, 3, 11, 11)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestUnsqueeze(unittest.TestCase):
    def test_unsqueeze_0(self):
        ts = [cranet.zeros((3, 4)) for _ in range(100)]
        z = cranet.stack(ts)
        self.assertTrue(z.shape == (100, 3, 4))


class TestChore(unittest.TestCase):
    def test_repr(self):
        print()
        a0 = np.random.rand(2, 2, 3)
        a = cranet.tensor(a0)
        b = torch.tensor(a0)
        # repr_a0 = repr(a0).replace("array", "tensor")
        print(a)
        print(b)
        print(repr(a0))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
