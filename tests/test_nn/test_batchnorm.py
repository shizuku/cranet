import os
import sys
import time
import unittest

import torch
from torch.nn import functional as t_F
from torch import nn as t_nn

from src import cranet
from src.cranet.nn import functional as c_F
from src.cranet import nn as c_nn

import numpy as np
from ..utils import teq


class TestBatchNorm2d(unittest.TestCase):
    def test_batch_norm_0(self):
        for _ in range(100):
            x = np.random.rand(1, 3, 4, 4)
            x_t = torch.tensor(x, requires_grad=True)
            x_c = cranet.tensor(x, requires_grad=True)
            w = np.random.rand(3)
            w_t = torch.tensor(w, requires_grad=True)
            w_c = cranet.tensor(w, requires_grad=True)
            b = np.random.rand(3)
            b_t = torch.tensor(b, requires_grad=True)
            b_c = cranet.tensor(b, requires_grad=True)
            x_mean = x.mean(axis=(0, 2, 3))
            x_mean_c = cranet.tensor(x_mean)
            x_mean_t = torch.tensor(x_mean)
            x_var = x.var(axis=(0, 2, 3))
            x_var_c = cranet.tensor(x_var)
            x_var_t = torch.tensor(x_var)
            y_t = t_F.batch_norm(x_t, x_mean_t, x_var_t, w_t, b_t, training=False, momentum=0, eps=1e-5)
            y_c = c_F.batch_norm(x_c, x_mean_c, x_var_c, w_c, b_c, training=False, momentum=0, eps=1e-5)
            self.assertTrue(teq(y_c, y_t, 1e-15))
            g = np.random.rand(1, 3, 4, 4)
            g_t = torch.tensor(g)
            g_c = cranet.tensor(g)
            y_t.backward(g_t)
            y_c.zero_grad()
            y_c.backward(g_c)
            self.assertTrue(teq(x_c.grad, x_t.grad, 1e-15))
            self.assertTrue(teq(w_c.grad, w_t.grad, 1e-10))
            self.assertTrue(teq(b_c.grad, b_t.grad, 1e-10))

    def test_batch_norm_1(self):
        for _ in range(100):
            x = np.random.rand(1, 3, 4, 4)
            x_t = torch.tensor(x, requires_grad=True)
            x_c = cranet.tensor(x, requires_grad=True)
            w = np.random.rand(3)
            w_t = torch.tensor(w, requires_grad=True)
            w_c = cranet.tensor(w, requires_grad=True)
            b = np.random.rand(3)
            b_t = torch.tensor(b, requires_grad=True)
            b_c = cranet.tensor(b, requires_grad=True)
            x_mean = x.mean(axis=(0, 2, 3))
            x_mean_c = cranet.tensor(x_mean)
            x_mean_t = torch.tensor(x_mean)
            x_var = x.var(axis=(0, 2, 3))
            x_var_c = cranet.tensor(x_var)
            x_var_t = torch.tensor(x_var)
            y_t = t_F.batch_norm(x_t, x_mean_t, x_var_t, w_t, b_t, training=False, momentum=0.1, eps=1e-5)
            y_c = c_F.batch_norm(x_c, x_mean_c, x_var_c, w_c, b_c, training=False, momentum=0.1, eps=1e-5)
            self.assertTrue(teq(y_c, y_t, 1e-15))
            g = np.random.rand(1, 3, 4, 4)
            g_t = torch.tensor(g)
            g_c = cranet.tensor(g)
            y_t.backward(g_t)
            y_c.zero_grad()
            y_c.backward(g_c)
            self.assertTrue(teq(x_c.grad, x_t.grad, 1e-15))
            self.assertTrue(teq(w_c.grad, w_t.grad, 1e-10))
            self.assertTrue(teq(b_c.grad, b_t.grad, 1e-10))

    def test_batch_norm_2(self):
        for _ in range(100):
            x = np.random.rand(1, 3, 4, 4)
            x_t = torch.tensor(x, requires_grad=True)
            x_c = cranet.tensor(x, requires_grad=True)
            w = np.random.rand(3)
            w_t = torch.tensor(w, requires_grad=True)
            w_c = cranet.tensor(w, requires_grad=True)
            b = np.random.rand(3)
            b_t = torch.tensor(b, requires_grad=True)
            b_c = cranet.tensor(b, requires_grad=True)
            x_mean = x.mean(axis=(0, 2, 3))
            x_mean_c = cranet.tensor(x_mean)
            x_mean_t = torch.tensor(x_mean)
            x_var = x.var(axis=(0, 2, 3))
            x_var_c = cranet.tensor(x_var)
            x_var_t = torch.tensor(x_var)
            y_t = t_F.batch_norm(x_t, x_mean_t, x_var_t, w_t, b_t, training=True, momentum=0, eps=1e-5)
            y_c = c_F.batch_norm(x_c, x_mean_c, x_var_c, w_c, b_c, training=True, momentum=0, eps=1e-5)
            self.assertTrue(teq(y_c, y_t, 1e-13))
            g = np.random.rand(1, 3, 4, 4)
            g_t = torch.tensor(g)
            g_c = cranet.tensor(g)
            y_t.backward(g_t)
            y_c.zero_grad()
            y_c.backward(g_c)
            self.assertTrue(teq(x_c.grad, x_t.grad, 1e-10))
            self.assertTrue(teq(w_c.grad, w_t.grad, 1e-10))
            self.assertTrue(teq(b_c.grad, b_t.grad, 1e-10))

    def test_batch_norm_3(self):
        back_c_time = 0
        back_t_time = 0
        for _ in range(100):
            x = np.random.rand(64, 3, 32, 32)
            x_t = torch.tensor(x, requires_grad=True)
            x_c = cranet.tensor(x, requires_grad=True)
            w = np.random.rand(3)
            w_t = torch.tensor(w, requires_grad=True)
            w_c = cranet.tensor(w, requires_grad=True)
            b = np.random.rand(3)
            b_t = torch.tensor(b, requires_grad=True)
            b_c = cranet.tensor(b, requires_grad=True)
            x_mean = x.mean(axis=(0, 2, 3))
            x_mean_c = cranet.tensor(x_mean)
            x_mean_t = torch.tensor(x_mean)
            x_var = x.var(axis=(0, 2, 3))
            x_var_c = cranet.tensor(x_var)
            x_var_t = torch.tensor(x_var)
            y_t = t_F.batch_norm(x_t, x_mean_t, x_var_t, w_t, b_t, training=True, momentum=0.1, eps=1e-5)
            y_c = c_F.batch_norm(x_c, x_mean_c, x_var_c, w_c, b_c, training=True, momentum=0.1, eps=1e-5)
            self.assertTrue(teq(y_c, y_t, 1e-10))
            g = np.random.rand(64, 3, 32, 32)
            g_t = torch.tensor(g)
            g_c = cranet.tensor(g)
            t_1 = time.time()
            y_t.backward(g_t)
            t2 = time.time()
            back_t_time += t2 - t_1
            c_1 = time.time()
            y_c.backward(g_c)
            c_2 = time.time()
            back_c_time += c_2 - c_1
            self.assertTrue(teq(x_c.grad, x_t.grad, 1e-10))
            self.assertTrue(teq(w_c.grad, w_t.grad, 1e-5))
            self.assertTrue(teq(b_c.grad, b_t.grad, 1e-5))
        print(back_t_time, back_c_time)

    def test_batch_norm_4(self):
        m_t = t_nn.BatchNorm2d(3, momentum=0, dtype=torch.float64)
        op_t = torch.optim.SGD(m_t.parameters(), 0.1)
        m_c = c_nn.BatchNorm2d(3, momentum=0)
        op_c = cranet.optim.SGD(m_c.parameters(), 0.1)
        m_t.train()
        m_c.train()
        for _ in range(100):
            x = np.random.rand(64, 3, 32, 32).astype(np.float64)
            x_t = torch.tensor(x, requires_grad=True)
            x_c = cranet.tensor(x, requires_grad=True)
            op_t.zero_grad()
            op_c.zero_grad()
            y_t = m_t(x_t)
            y_c = m_c(x_c)
            self.assertTrue(teq(y_t, y_c, 1e-8))
            g = np.random.rand(64, 3, 32, 32).astype(np.float64)
            y_t.backward(torch.tensor(g))
            y_c.backward(cranet.tensor(g))
            op_t.step()
            op_c.step()
            self.assertTrue(teq(x_t.grad, x_c.grad, 1e-8))

    def test_batch_norm_5(self):
        m_t = t_nn.BatchNorm2d(3, momentum=0.1, dtype=torch.float64)
        op_t = torch.optim.SGD(m_t.parameters(), 0.1)
        m_c = c_nn.BatchNorm2d(3, momentum=0.1)
        op_c = cranet.optim.SGD(m_c.parameters(), 0.1)
        m_t.train()
        m_c.train()
        for _ in range(100):
            x = np.random.rand(64, 3, 32, 32).astype(np.float64)
            x_t = torch.tensor(x, requires_grad=True)
            x_c = cranet.tensor(x, requires_grad=True)
            op_t.zero_grad()
            op_c.zero_grad()
            y_t = m_t(x_t)
            y_c = m_c(x_c)
            self.assertTrue(teq(y_t, y_c, 1e-8))
            g = np.random.rand(64, 3, 32, 32).astype(np.float64)
            y_t.backward(torch.tensor(g))
            y_c.backward(cranet.tensor(g))
            op_t.step()
            op_c.step()
            self.assertTrue(teq(x_t.grad, x_c.grad, 1e-8))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
