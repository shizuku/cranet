import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.dpln.nn import functional as dpln_F

from src import dpln


def np_feq(a: np.ndarray, b: np.ndarray, epsilon: float = 2e-15) -> bool:
    return (np.abs(a - b) < epsilon).all()


class TestFunction(unittest.TestCase):
    def test_mse(self):
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

    def cross_entropy(self):
        pass
