import os
import sys
import unittest

import numpy as np
import torch

from src.dpln.autograd import Tensor, ones_like


class TestTensor(unittest.TestCase):
    def setUp(self):
        self.t1 = Tensor([1., 2., 3.], requires_grad=True)
        self.t2 = Tensor([4., 5., 6.], requires_grad=True)

    def test_sum(self):
        self.t1.zero_grad()
        sum_t1 = self.t1.sum()
        self.assertEqual(sum_t1, Tensor(6))
        sum_t1.backward()
        self.assertEqual(self.t1.grad.data.tolist(), [1, 1, 1])

        self.t1.zero_grad()
        sum_t1.backward(Tensor(3))
        self.assertEqual(self.t1.grad.data.tolist(), [3, 3, 3])

    def test_add(self):
        self.t1.zero_grad()
        self.t2.zero_grad()
        sum = self.t1 + self.t2
        self.assertEqual(sum, Tensor([5, 7, 9]))
        sum.backward(Tensor([1, 1, 1]))
        self.assertEqual(self.t1.grad, Tensor([1, 1, 1]))
        self.assertEqual(self.t2.grad, Tensor([1, 1, 1]))

    def test_sub(self):
        self.t1.zero_grad()
        self.t2.zero_grad()
        x = self.t1 - self.t2
        self.assertEqual(x, Tensor([-3, -3, -3]))
        x.backward(Tensor([1, 1, 1]))
        self.assertEqual(self.t1.grad, Tensor([1, 1, 1]))
        self.assertEqual(self.t2.grad, Tensor([-1, -1, -1]))

    def test_neg(self):
        self.t1.zero_grad()
        n = -self.t1
        self.assertEqual(n, Tensor([-1, -2, -3]))
        n.backward(Tensor([1, 1, 1]))
        self.assertEqual(self.t1.grad, Tensor([-1, -1, -1]))

    def test_mul(self):
        self.t1.zero_grad()
        self.t2.zero_grad()
        x = self.t1 * self.t2
        self.assertEqual(x, Tensor([4, 10, 18]))
        x.backward(Tensor([1, 1, 1]))
        self.assertEqual(self.t1.grad, self.t2)
        self.assertEqual(self.t2.grad, self.t1)

    def test_truediv(self):
        a = Tensor([1., 2., 1.], requires_grad=True)
        b = Tensor([4.], requires_grad=True)
        y = b / a
        self.assertEqual(y, Tensor([4., 2., 4.]))
        y.backward(Tensor([1., 1., 1.]))
        self.assertEqual(a.grad, Tensor([-4., -1., -4.]))
        self.assertEqual(b.grad, Tensor([2.5]))

    def test_matmul(self):
        for _ in range(1000):
            a = np.random.rand(8, 10)
            b = np.random.rand(10, 8)
            a0 = Tensor(a, requires_grad=True)
            b0 = Tensor(b, requires_grad=True)
            x = a0 @ b0
            x.backward(ones_like(x))
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
            a0 = Tensor(a, requires_grad=True)
            b0 = Tensor(b, requires_grad=True)
            x = a0 @ b0
            x.backward(ones_like(x))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.tensor(b, requires_grad=True)
            y = a1 @ b1
            y.backward(torch.ones_like(y))
            self.assertTrue((x.numpy() == y.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.numpy()).all())
            self.assertTrue((b0.grad.numpy() == b1.grad.numpy()).all())

    def test_transpose(self):
        for _ in range(100):
            a = np.random.rand(3, 4)
            a0 = Tensor(a, requires_grad=True)
            b0 = a0.transpose(0, 1)
            b0.backward(ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 0, 1)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(3, 4, 5, 6)
            a0 = Tensor(a, requires_grad=True)
            b0 = a0.transpose(1, 3)
            b0.backward(ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 1, 3)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(100):
            a = np.random.rand(7, 9, 2, 4, 8, 3)
            a0 = Tensor(a, requires_grad=True)
            b0 = a0.transpose(2, 0)
            b0.backward(ones_like(b0))
            a1 = torch.tensor(a, requires_grad=True)
            b1 = torch.transpose(a1, 2, 0)
            b1.backward(torch.ones_like(b1))
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

    def test_permute(self):
        # for _ in range(10):
        #     a = np.random.rand(16)
        #     a0 = Tensor(a, requires_grad=True)
        #     b0 = a0.permute([0])
        #     a1 = torch.tensor(a, requires_grad=True)
        #     b1 = torch.permute(a1, [0])
        #     self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
        #     b0.backward(ones_like(b0))
        #     b1.backward(torch.ones_like(b1))
        #     self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
        #
        # for _ in range(10):
        #     a = np.random.rand(3, 4)
        #     a0 = Tensor(a, requires_grad=True)
        #     b0 = a0.permute([1, 0])
        #     a1 = torch.tensor(a, requires_grad=True)
        #     b1 = torch.permute(a1, [1, 0])
        #     self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
        #     b0.backward(ones_like(b0))
        #     b1.backward(torch.ones_like(b1))
        #     self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
        #
        # for _ in range(10):
        #     a = np.random.rand(3, 4, 5, 6)
        #     a0 = Tensor(a, requires_grad=True)
        #     b0 = a0.permute([0, 1, 3, 2])
        #     b0.backward(ones_like(b0))
        #     a1 = torch.tensor(a, requires_grad=True)
        #     b1 = torch.permute(a1, [0, 1, 3, 2])
        #     b1.backward(torch.ones_like(b1))
        #     self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
        #     self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())
        #
        # for _ in range(10):
        #     a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10)
        #     a0 = Tensor(a, requires_grad=True)
        #     b0 = a0.permute([7, 6, 5, 4, 3, 2, 1, 0])
        #     b0.backward(ones_like(b0))
        #     a1 = torch.tensor(a, requires_grad=True)
        #     b1 = torch.permute(a1, [7, 6, 5, 4, 3, 2, 1, 0])
        #     b1.backward(torch.ones_like(b1))
        #     self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
        #     self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())

        for _ in range(1):
            a = np.random.rand(3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            a0 = Tensor(a.copy(), requires_grad=True)
            b0 = a0.permute([9, 1, 7, 6, 5, 3, 4, 2, 8, 0])
            a1 = torch.tensor(a.copy(), requires_grad=True)
            b1 = torch.permute(a1, [9, 1, 7, 6, 5, 3, 4, 2, 8, 0])
            self.assertTrue((b0.numpy() == b1.detach().numpy()).all())
            b1.backward(torch.ones_like(b1))
            b0.backward(ones_like(b0))
            self.assertTrue((a0.grad.numpy() == a1.grad.detach().numpy()).all())


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
