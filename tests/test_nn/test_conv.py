import os
import sys
import time
import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.cranet.nn import functional as cranet_F

from src import cranet

from ..utils import teq


class TestConv2d(unittest.TestCase):
    # TODO: FIX ERROR
    def test_conv2d_1(self):
        for _ in range(100):
            w = np.random.rand(1, 1, 3, 3)
            b = np.random.rand(1)
            x = np.random.rand(1, 1, 7, 7)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = cranet.Tensor(w, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            x_d = cranet.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, padding=1)
            y_d = cranet_F.conv2d(x_d, w_d, b_d, padding=1)
            self.assertTrue(teq(y_t, y_d, 1e-13))
            grad = np.random.rand(1, 1, 7, 7)
            grad_t = torch.tensor(grad)
            grad_d = cranet.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(teq(w_t.grad, w_d.grad, 1e-10))
            self.assertTrue(teq(b_t.grad, b_d.grad, 1e-10))
            self.assertTrue(teq(x_t.grad, x_d.grad, 1e-10))

    def test_conv2d_2(self):
        time_t = 0
        time_d = 0
        for _ in range(100):
            w = np.random.rand(32, 3, 3, 3)
            b = np.random.rand(32)
            x = np.random.rand(64, 3, 32, 32)
            grad = np.random.rand(64, 32, 32, 32)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = cranet.Tensor(w, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            x_d = cranet.Tensor(x, requires_grad=True)
            grad_t = torch.tensor(grad)
            grad_d = cranet.Tensor(grad)
            beg_t = time.time()
            y_t = torch_F.conv2d(x_t, w_t, b_t, padding=1)
            y_t.backward(grad_t)
            end_t = time.time()
            time_t += end_t - beg_t
            beg_d = time.time()
            y_d = cranet_F.conv2d(x_d, w_d, b_d, padding=1)
            y_d.backward(grad_d)
            end_d = time.time()
            time_d += end_d - beg_d
            self.assertTrue(teq(y_t, y_d, 1e-13))
            self.assertTrue(teq(w_t.grad, w_d.grad, 1e-8))
            self.assertTrue(teq(b_t.grad, b_d.grad, 1e-10))
            self.assertTrue(teq(x_t.grad, x_d.grad, 1e-10))
        print("torch", time_t)
        print("cranet", time_d)

    def test_conv2d_2_2(self):
        for _ in range(100):
            w = np.random.rand(64, 32, 3, 3)
            b = np.random.rand(64)
            x = np.random.rand(2, 32, 16, 16)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = cranet.Tensor(w, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            x_d = cranet.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, padding=1)
            y_d = cranet_F.conv2d(x_d, w_d, b_d, padding=1)
            self.assertTrue(teq(y_t, y_d, 1e-10))
            grad = np.random.rand(2, 64, 16, 16)
            grad_t = torch.tensor(grad)
            grad_d = cranet.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(teq(w_t.grad, w_d.grad, 1e-10))
            self.assertTrue(teq(b_t.grad, b_d.grad, 1e-10))
            self.assertTrue(teq(x_t.grad, x_d.grad, 1e-10))

    # def test_conv2d_3(self):
    #     for _ in range(100):
    #         w = np.random.rand(8, 3, 7, 5)
    #         b = np.random.rand(8)
    #         x = np.random.rand(2, 3, 64, 32)
    #         w_t = torch.tensor(w, requires_grad=True)
    #         b_t = torch.tensor(b, requires_grad=True)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         w_d = cranet.Tensor(w, requires_grad=True)
    #         b_d = cranet.Tensor(b, requires_grad=True)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_t = torch_F.conv2d(x_t, w_t, b_t, stride=2, padding=4)
    #         y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=2, padding=4)
    #         self.assertTrue(teq(y_t, y_d, 1e-13))
    #         grad = np.random.rand(2, 8, 33, 18)
    #         grad_t = torch.tensor(grad)
    #         grad_d = cranet.Tensor(grad)
    #         y_t.backward(grad_t)
    #         y_d.backward(grad_d)
    #         self.assertTrue(teq(w_t.grad, w_d.grad, 1e-8))
    #         self.assertTrue(teq(b_t.grad, b_d.grad, 1e-8))
    #         self.assertTrue(teq(x_t.grad, x_d.grad, 1e-8))

    # def test_conv2d_4(self):
    #     for _ in range(100):
    #         w = np.random.rand(8, 3, 7, 9)
    #         b = np.random.rand(8)
    #         x = np.random.rand(2, 3, 64, 32)
    #         w_t = torch.tensor(w, requires_grad=True)
    #         b_t = torch.tensor(b, requires_grad=True)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         w_d = cranet.Tensor(w, requires_grad=True)
    #         b_d = cranet.Tensor(b, requires_grad=True)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_t = torch_F.conv2d(x_t, w_t, b_t, stride=[2, 1], padding=(5, 6))
    #         y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=[2, 1], padding=(5, 6))
    #         self.assertTrue(teq(y_t, y_d, 2e-13))
    #         grad = np.random.rand(2, 8, 34, 36)
    #         grad_t = torch.tensor(grad)
    #         grad_d = cranet.Tensor(grad)
    #         y_t.backward(grad_t)
    #         y_d.backward(grad_d)
    #         self.assertTrue(teq(w_t.grad, w_d.grad, 2e-10))
    #         self.assertTrue(teq(b_t.grad, b_d.grad, 2e-10))
    #         self.assertTrue(teq(x_t.grad, x_d.grad, 2e-10))

    # def test_conv2d_5(self):
    #     for _ in range(10):
    #         w = np.random.rand(8, 3, 7, 9)
    #         b = np.random.rand(8)
    #         x = np.random.rand(2, 3, 64, 32)
    #         w_t = torch.tensor(w, requires_grad=True)
    #         b_t = torch.tensor(b, requires_grad=True)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         w_d = cranet.Tensor(w, requires_grad=True)
    #         b_d = cranet.Tensor(b, requires_grad=True)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_t = torch_F.conv2d(x_t, w_t, b_t, stride=(2, 4), padding=(9, 1))
    #         y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=(2, 4), padding=(9, 1))
    #         self.assertTrue(teq(y_t, y_d, 2e-13))
    #         grad = np.random.rand(2, 8, 38, 7)
    #         grad_t = torch.tensor(grad)
    #         grad_d = cranet.Tensor(grad)
    #         y_t.backward(grad_t)
    #         y_d.backward(grad_d)
    #         self.assertTrue(teq(w_t.grad, w_d.grad, 2e-10))
    #         self.assertTrue(teq(b_t.grad, b_d.grad, 2e-10))
    #         self.assertTrue(teq(x_t.grad, x_d.grad, 2e-10))

    # def test_conv2d_6(self):
    #     for _ in range(10):
    #         w = np.random.rand(8, 3, 3, 3)
    #         b = np.random.rand(8)
    #         x = np.random.rand(2, 3, 32, 32)
    #         w_t = torch.tensor(w, requires_grad=True)
    #         b_t = torch.tensor(b, requires_grad=True)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         w_d = cranet.Tensor(w, requires_grad=True)
    #         b_d = cranet.Tensor(b, requires_grad=True)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_t = torch_F.conv2d(x_t, w_t, b_t, stride=(2, 2), padding=1, dilation=2)
    #         y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=(2, 2), padding=1, dilation=2)
    #         self.assertTrue(teq(y_t, y_d, 2e-13))
    #         grad = np.random.rand(2, 8, 15, 15)
    #         grad_t = torch.tensor(grad)
    #         grad_d = cranet.Tensor(grad)
    #         y_t.backward(grad_t)
    #         y_d.backward(grad_d)
    #         self.assertTrue(teq(w_t.grad, w_d.grad, 2e-10))
    #         self.assertTrue(teq(b_t.grad, b_d.grad, 2e-10))
    #         self.assertTrue(teq(x_t.grad, x_d.grad, 2e-10))

    # def test_conv2d_7(self):
    #     for _ in range(10):
    #         w = np.random.rand(8, 3, 3, 3)
    #         b = np.random.rand(8)
    #         x = np.random.rand(2, 3, 35, 28)
    #         w_t = torch.tensor(w, requires_grad=True)
    #         b_t = torch.tensor(b, requires_grad=True)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         w_d = cranet.Tensor(w, requires_grad=True)
    #         b_d = cranet.Tensor(b, requires_grad=True)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_t = torch_F.conv2d(x_t, w_t, b_t, stride=(2, 2), padding=1, dilation=2)
    #         y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=(2, 2), padding=1, dilation=2)
    #         self.assertTrue(teq(y_t, y_d, 2e-13))
    #         grad = np.random.rand(2, 8, 17, 13)
    #         grad_t = torch.tensor(grad)
    #         grad_d = cranet.Tensor(grad)
    #         y_t.backward(grad_t)
    #         y_d.backward(grad_d)
    #         self.assertTrue(teq(w_t.grad, w_d.grad, 2e-10))
    #         self.assertTrue(teq(b_t.grad, b_d.grad, 2e-10))
    #         self.assertTrue(teq(x_t.grad, x_d.grad, 2e-10))

    def test_conv2d_8(self):
        for _ in range(100):
            w = np.random.rand(8, 3, 3, 3)
            b = np.random.rand(8)
            x = np.random.rand(1, 6, 7, 7)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = cranet.Tensor(w, requires_grad=True)
            b_d = cranet.Tensor(b, requires_grad=True)
            x_d = cranet.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, stride=1, padding=1, groups=2)
            y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=1, padding=1, groups=2)
            self.assertTrue(teq(y_t, y_d, 2e-13))
            grad = np.random.rand(1, 8, 7, 7)
            grad_t = torch.tensor(grad)
            grad_d = cranet.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(teq(w_t.grad, w_d.grad, 2e-10))
            self.assertTrue(teq(b_t.grad, b_d.grad, 2e-10))
            self.assertTrue(teq(x_t.grad, x_d.grad, 2e-10))

    # def test_conv2d_9(self):
    #     for _ in range(1000):
    #         w = np.random.rand(16, 4, 3, 3)
    #         b = np.random.rand(16)
    #         x = np.random.rand(1, 8, 32, 32)
    #         w_t = torch.tensor(w, requires_grad=True)
    #         b_t = torch.tensor(b, requires_grad=True)
    #         x_t = torch.tensor(x, requires_grad=True)
    #         w_d = cranet.Tensor(w, requires_grad=True)
    #         b_d = cranet.Tensor(b, requires_grad=True)
    #         x_d = cranet.Tensor(x, requires_grad=True)
    #         y_t = torch_F.conv2d(x_t, w_t, b_t, stride=2, padding=1, groups=2)
    #         y_d = cranet_F.conv2d(x_d, w_d, b_d, stride=2, padding=1, groups=2)
    #         self.assertTrue(teq(y_t, y_d, 2e-13))
    #         grad = np.random.rand(1, 16, 16, 16)
    #         grad_t = torch.tensor(grad)
    #         grad_d = cranet.Tensor(grad)
    #         y_t.backward(grad_t)
    #         y_d.backward(grad_d)
    #         self.assertTrue(teq(w_t.grad, w_d.grad, 2e-10))
    #         self.assertTrue(teq(b_t.grad, b_d.grad, 2e-10))
    #         self.assertTrue(teq(x_t.grad, x_d.grad, 2e-10))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
