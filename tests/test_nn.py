import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.dpln.nn import functional as dpln_F

from src import dpln


def np_feq(a: np.ndarray, b: np.ndarray, epsilon: float = 2e-15) -> bool:
    return (np.abs(a - b) < epsilon).all()


class TestConv2d(unittest.TestCase):
    def test_con2d_0(self):
        model = dpln.nn.Conv2d(3, 4, 3, stride=2, padding='same')
        sample_inp = dpln.uniform((1, 3, 32, 32))
        sample_out = model(sample_inp)
        self.assertTrue(sample_out.shape == (1, 4, 16, 16))

    def test_conv2d_1(self):
        for _ in range(100):
            w = np.random.rand(8, 1, 3, 3)
            b = np.random.rand(8)
            x = np.random.rand(1, 1, 7, 7)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = dpln.Tensor(w, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            x_d = dpln.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, stride=1, padding=1)
            y_d = dpln_F.conv2d(x_d, w_d, b_d, stride=1, padding=1)
            self.assertTrue(np_feq(y_t.detach().numpy(), y_d.numpy(), 2e-13))
            grad = np.random.rand(1, 8, 7, 7)
            grad_t = torch.tensor(grad)
            grad_d = dpln.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(np_feq(w_t.detach().numpy(), w_d.numpy()))
            self.assertTrue(np_feq(b_t.detach().numpy(), b_d.numpy()))
            self.assertTrue(np_feq(x_t.detach().numpy(), x_d.numpy()))

    def test_conv2d_2(self):
        for _ in range(100):
            w = np.random.rand(8, 3, 5, 5)
            b = np.random.rand(8)
            x = np.random.rand(2, 3, 32, 32)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = dpln.Tensor(w, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            x_d = dpln.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, stride=1, padding=2)
            y_d = dpln_F.conv2d(x_d, w_d, b_d, stride=1, padding=2)
            self.assertTrue(np_feq(y_t.detach().numpy(), y_d.numpy(), 2e-13))
            grad = np.random.rand(2, 8, 32, 32)
            grad_t = torch.tensor(grad)
            grad_d = dpln.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(np_feq(w_t.detach().numpy(), w_d.numpy()))
            self.assertTrue(np_feq(b_t.detach().numpy(), b_d.numpy()))
            self.assertTrue(np_feq(x_t.detach().numpy(), x_d.numpy()))

    def test_conv2d_3(self):
        for _ in range(100):
            w = np.random.rand(8, 3, 7, 5)
            b = np.random.rand(8)
            x = np.random.rand(2, 3, 64, 32)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = dpln.Tensor(w, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            x_d = dpln.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, stride=2, padding=4)
            y_d = dpln_F.conv2d(x_d, w_d, b_d, stride=2, padding=4)
            self.assertTrue(np_feq(y_t.detach().numpy(), y_d.numpy(), 2e-13))
            grad = np.random.rand(2, 8, 33, 18)
            grad_t = torch.tensor(grad)
            grad_d = dpln.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(np_feq(w_t.detach().numpy(), w_d.numpy()))
            self.assertTrue(np_feq(b_t.detach().numpy(), b_d.numpy()))
            self.assertTrue(np_feq(x_t.detach().numpy(), x_d.numpy()))

    def test_conv2d_4(self):
        for _ in range(100):
            w = np.random.rand(8, 3, 7, 9)
            b = np.random.rand(8)
            x = np.random.rand(2, 3, 64, 32)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = dpln.Tensor(w, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            x_d = dpln.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, stride=[2, 1], padding=(5, 6))
            y_d = dpln_F.conv2d(x_d, w_d, b_d, stride=[2, 1], padding=(5, 6))
            self.assertTrue(np_feq(y_t.detach().numpy(), y_d.numpy(), 2e-13))
            grad = np.random.rand(2, 8, 34, 36)
            grad_t = torch.tensor(grad)
            grad_d = dpln.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(np_feq(w_t.detach().numpy(), w_d.numpy()))
            self.assertTrue(np_feq(b_t.detach().numpy(), b_d.numpy()))
            self.assertTrue(np_feq(x_t.detach().numpy(), x_d.numpy()))

    def test_conv2d_5(self):
        for _ in range(100):
            w = np.random.rand(8, 3, 7, 9)
            b = np.random.rand(8)
            x = np.random.rand(2, 3, 64, 32)
            w_t = torch.tensor(w, requires_grad=True)
            b_t = torch.tensor(b, requires_grad=True)
            x_t = torch.tensor(x, requires_grad=True)
            w_d = dpln.Tensor(w, requires_grad=True)
            b_d = dpln.Tensor(b, requires_grad=True)
            x_d = dpln.Tensor(x, requires_grad=True)
            y_t = torch_F.conv2d(x_t, w_t, b_t, stride=(2, 4), padding=(9, 1))
            y_d = dpln_F.conv2d(x_d, w_d, b_d, stride=(2, 4), padding=(9, 1))
            self.assertTrue(np_feq(y_t.detach().numpy(), y_d.numpy(), 2e-13))
            grad = np.random.rand(2, 8, 38, 7)
            grad_t = torch.tensor(grad)
            grad_d = dpln.Tensor(grad)
            y_t.backward(grad_t)
            y_d.backward(grad_d)
            self.assertTrue(np_feq(w_t.detach().numpy(), w_d.numpy()))
            self.assertTrue(np_feq(b_t.detach().numpy(), b_d.numpy()))
            self.assertTrue(np_feq(x_t.detach().numpy(), x_d.numpy()))
