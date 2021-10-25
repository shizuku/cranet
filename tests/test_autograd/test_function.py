import os
import sys
import unittest

import numpy as np
import torch

from src import dpln

from ..utils import np_feq


class TestAbs(unittest.TestCase):
    def test_abs_0(self):
        for _ in range(100):
            shape = [6]
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = dpln.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_abs_1(self):
        for _ in range(100):
            shape = (2, 2)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = dpln.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_abs_2(self):
        for _ in range(100):
            shape = (2, 3, 4, 5, 7, 9)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = dpln.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestMax(unittest.TestCase):
    def test_max_0(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = dpln.max(a_d)
            b_t = torch.max(a_t)
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            b_t.backward()
            b_d.backward()
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))

    def test_max_1(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = dpln.max(a_d)
            b_t = torch.max(a_t)
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand()
            b_t.backward(torch.tensor(g))
            b_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy(), 2e-7))

    def test_max_5(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = dpln.max(a_d, dim=axis)
            b_t = torch.max(a_t, dim=axis).values
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))

    def test_max_6(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = dpln.max(a_d, dim=axis, keepdim=True)
            b_t = torch.max(a_t, dim=axis, keepdim=True).values
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))


class TestMin(unittest.TestCase):
    def test_min_0(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = dpln.min(a_d)
            b_t = torch.min(a_t)
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            b_t.backward()
            b_d.backward()
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))

    def test_min_1(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = dpln.min(a_d)
            b_t = torch.min(a_t)
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand()
            b_t.backward(torch.tensor(g))
            b_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy(), 2e-7))

    def test_min_5(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = dpln.min(a_d, dim=axis)
            b_t = torch.min(a_t, dim=axis).values
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))

    def test_min_6(self):
        for _ in range(100):
            a = np.random.rand(5, 6, 7, 8, 9)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            axis = np.random.randint(0, 5)
            b_d = dpln.min(a_d, dim=axis, keepdim=True)
            b_t = torch.min(a_t, dim=axis, keepdim=True).values
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand(*b_d.shape)
            b_t.backward(torch.tensor(g))
            b_d.backward(dpln.Tensor(g))
            self.assertTrue(np_feq(a_d.grad.numpy(), a_t.grad.detach().numpy()))


class TestLog(unittest.TestCase):
    def test_log_0(self):
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

    def test_log_1(self):
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


class TestExp(unittest.TestCase):
    def test_exp_0(self):
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

    def test_exp_1(self):
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


class TestRelu(unittest.TestCase):
    def test_relu(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.relu(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.relu(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.sigmoid(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.sigmoid(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.softmax(a0, dim=-1)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.softmax(a1, dim=-1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestTanh(unittest.TestCase):
    def test_tanh(self):
        for _ in range(100):
            a = np.random.rand(13, 10)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.tanh(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.tanh(a1)
            self.assertTrue(np_feq(b0.numpy(), b1.detach().numpy()))
            b1.backward(torch.ones_like(b1) * 0.1)
            b0.backward(dpln.ones_like(b0) * 0.1)
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))


class TestFlatten(unittest.TestCase):
    def test_flatten(self):
        for _ in range(100):
            a = np.random.rand(2, 2, 2)
            a_d = dpln.Tensor(a, requires_grad=True)
            a_t = torch.tensor(a, requires_grad=True)
            b_d = dpln.flatten(a_d)
            b_t = torch.flatten(a_t)
            self.assertTrue(np_feq(b_d.numpy(), b_t.detach().numpy()))
            g = np.random.rand(2 * 2 * 2)
            g_d = dpln.Tensor(g)
            g_t = torch.tensor(g)
            b_d.backward(g_d)
            b_t.backward(g_t)
            self.assertTrue(np_feq(a_d.numpy(), a_t.detach().numpy()))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
