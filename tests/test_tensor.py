import os
import random
import sys
import unittest

import numpy as np
import torch

from src import dpln


class TestTensor(unittest.TestCase):
    def setUp(self):
        pass

    def test_sum(self):
        for _ in range(100):
            a = np.random.rand(100)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.sum()
            b1 = torch.sum(a1)
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-13).all())
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = dpln.Tensor(a, requires_grad=True)
            a1 = torch.tensor(a, requires_grad=True)
            b0 = a0.sum()
            b1 = torch.sum(a1)
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-13).all(), f"{b0.numpy()}\n{b1.detach().numpy()}")
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

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
            c0 = a0 + b0
            c1 = a1 + b1
            c0.backward(dpln.ones_like(c0))
            c1.backward(torch.ones_like(c1))
            self.assertTrue((c0.numpy() == c1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.detach().numpy()).all())

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
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-15).all())
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() - a1.grad.detach().numpy() < 2e-15).all())

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
            self.assertTrue((c0.numpy() - c1.detach().numpy() < 2e-15).all())
            self.assertTrue((a0.grad.numpy() - a1.grad.detach().numpy() < 2e-10).all())
            self.assertTrue((b0.grad.numpy() - b1.grad.detach().numpy() < 2e-10).all())

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
            self.assertTrue((c0.numpy() - c1.detach().numpy() < 2e-15).all())
            self.assertTrue((a0.grad.numpy() - a1.grad.detach().numpy() < 2e-10).all())
            self.assertTrue((b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7).all(),
                            f"{b0.grad.numpy() - b1.grad.detach().numpy()}{b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7}{np.argmin(b0.grad.numpy() - b1.grad.detach().numpy() < 2e-7)}")

    def test_matmul(self):
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

        for _ in range(1000):
            a = np.random.rand(16, 16, 8, 10)
            b = np.random.rand(16, 16, 10, 8)
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

    def test_log(self):
        for _ in range(100):
            a = np.random.rand(3)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.log(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.log(a1)
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-15).all(), f"{b0},{b1}")
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.log(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.log(a1)
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-15).all(), f"{b0},{b1}")
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
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-15).all())
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() - a1.grad.detach().numpy() < 2e-15).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5)
            a0 = dpln.Tensor(a, requires_grad=True)
            b0 = dpln.exp(a0)
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.exp(a1)
            self.assertTrue((b0.numpy() - b1.detach().numpy() < 2e-15).all())
            b0.backward(dpln.ones_like(b0))
            b1.backward(torch.ones_like(b1))
            self.assertTrue((a0.grad.numpy() - a1.grad.detach().numpy() < 2e-15).all())


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
