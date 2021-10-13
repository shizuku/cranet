import os
import sys
import unittest

from src.dpln.autograd import Tensor, add


class TestTensorSum(unittest.TestCase):
    def setUp(self):
        self.t1 = Tensor([1, 2, 3], requires_grad=True)
        self.t2 = Tensor([4, 5, 6], requires_grad=True)

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
        sum = add(self.t1, self.t2)
        self.assertEqual(sum, Tensor([5, 7, 9]))
        sum.backward(Tensor([1, 1, 1]))
        self.assertEqual(self.t1.grad, Tensor(1))
        self.assertEqual(self.t2.grad, Tensor(1))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
