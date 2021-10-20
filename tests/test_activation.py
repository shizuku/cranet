import unittest

import numpy as np
import torch

from dpln.autograd import function as AF
from torch.nn import functional as torch_F
from src.dpln.nn import functional as dpln_F

from src import dpln


def np_feq(a: np.ndarray, b: np.ndarray, epsilon: float = 2e-15) -> bool:
    return (np.abs(a - b) < epsilon).all()


class TestFunction(unittest.TestCase):
    def test_relu(self):
        for _ in range(100):
            shape = (4)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = dpln_F.relu(a0)
            c1 = torch_F.relu(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            shape = (2, 2)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = dpln_F.relu(a0)
            c1 = torch_F.relu(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            shape = (2, 3, 4, 5, 7, 9)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = dpln_F.relu(a0)
            c1 = torch_F.relu(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

    def test_abs(self):
        for _ in range(100):
            shape = (6)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = AF.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            shape = (2, 2)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = AF.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))

        for _ in range(100):
            shape = (2, 3, 4, 5, 7, 9)
            a = np.random.uniform(-1, 1, shape)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)

            c0 = AF.abs(a0)
            c1 = torch.abs(a1)
            delta = np.random.uniform(-1, 1, shape)
            delta0 = dpln.Tensor(delta)
            delta1 = torch.tensor(delta)
            c0.zero_grad()
            c0.backward(delta0)
            c1.backward(delta1)
            self.assertTrue(np_feq(c0.numpy(), c1.detach().numpy()))
            self.assertTrue(np_feq(a0.grad.numpy(), a1.grad.detach().numpy()))
