import os
import sys
import unittest

import numpy as np
import torch

from src import cranet

from ..utils import teq


class TestAbs(unittest.TestCase):
    def test_abs_0(self):
        for _ in range(100):
            shape = [6]
            a = np.random.uniform(-1, 1, shape)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = cranet.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = cranet.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))

    def test_abs_1(self):
        for _ in range(100):
            shape = (2, 2)
            a = np.random.uniform(-1, 1, shape)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = cranet.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = cranet.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))

    def test_abs_2(self):
        for _ in range(100):
            shape = (2, 3, 4, 5, 7, 9)
            a = np.random.uniform(-1, 1, shape)
            a0 = cranet.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = cranet.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = cranet.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(teq(c0, c1))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestMax(unittest.TestCase):
    def test_max_0(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = cranet.max(a_d)
            b_t = torch.max(a_t)
            self.assertTrue(teq(b_d, b_t))
            b_t.backward()
            b_d.backward()
            self.assertTrue(teq(a_d.grad, a_t.grad))

    def test_max_1(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = cranet.max(a_d)
            b_t = torch.max(a_t)
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand()
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 2e-7))

    def test_max_5(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = cranet.max(a_d, dim=axis)
            b_t = torch.max(a_t, dim=axis).values
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad))

    def test_max_6(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = cranet.max(a_d, dim=axis, keepdim=True)
            b_t = torch.max(a_t, dim=axis, keepdim=True).values
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad))


class TestMin(unittest.TestCase):
    def test_min_0(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = cranet.min(a_d)
            b_t = torch.min(a_t)
            self.assertTrue(teq(b_d, b_t))
            b_t.backward()
            b_d.backward()
            self.assertTrue(teq(a_d.grad, a_t.grad))

    def test_min_1(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = cranet.min(a_d)
            b_t = torch.min(a_t)
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand()
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad, 2e-7))

    def test_min_5(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = cranet.min(a_d, dim=axis)
            b_t = torch.min(a_t, dim=axis).values
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad))

    def test_min_6(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = cranet.min(a_d, dim=axis, keepdim=True)
            b_t = torch.min(a_t, dim=axis, keepdim=True).values
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(cranet.Tensor(g))
            self.assertTrue(teq(a_d.grad, a_t.grad))


class TestLog(unittest.TestCase):
    def test_log_0(self):
        for _ in range(100):
            a = np.random.rand(3)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.log(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.log(a1)
            self.assertTrue(teq(b0, b1), f"{b0},{b1}")
            g = np.random.rand(3)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue((a0.grad, a1.grad, 1e-15))

    def test_log_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.log(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.log(a1)
            self.assertTrue(teq(b0, b1), f"{b0},{b1}")
            g = np.random.rand(3, 4, 5)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue((a0.grad, a1.grad), 1e-15)


class TestExp(unittest.TestCase):
    def test_exp_0(self):
        for _ in range(100):
            a = np.random.rand(8)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.exp(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.exp(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(8)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))

    def test_exp_1(self):
        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.exp(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.exp(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(3, 4, 5)
            b0.backward(cranet.tensor(g))
            b1.backward(torch.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestRelu(unittest.TestCase):
    def test_relu(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.relu(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.relu(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(13, 10)
            b1.backward(torch.tensor(g))
            b0.backward(cranet.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.sigmoid(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.sigmoid(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(13, 10)
            b1.backward(torch.tensor(g))
            b0.backward(cranet.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        for _ in range(100):
            a = np.random.rand(64, 10)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.softmax(a0, dim=-1)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.softmax(a1, dim=-1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(64, 10)
            b1.backward(torch.tensor(g))
            b0.backward(cranet.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestTanh(unittest.TestCase):
    def test_tanh(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = cranet.Tensor(a, requires_grad=True)
            b0 = cranet.tanh(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.tanh(a1)
            self.assertTrue(teq(b0, b1))
            g = np.random.rand(13, 10)
            b1.backward(torch.tensor(g))
            b0.backward(cranet.tensor(g))
            self.assertTrue(teq(a0.grad, a1.grad))


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        for _ in range(100):
            a = np.random.rand(2, 2, 2)
            a_d = cranet.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = cranet.flatten(a_d)
            b_t = torch.flatten(a_t)
            self.assertTrue(teq(b_d, b_t))
            g = np.random.rand(2 * 2 * 2)
            g_d = cranet.Tensor(g)
            g_t = torch.tensor(g)
            b_d.backward(g_d)
            b_t.backward(g_t)
            self.assertTrue(teq(a_d, a_t))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
