import os
import sys
import unittest

import numpy as np
import torch

from torch.nn import functional as torch_F
from src.dpln.nn import functional as dpln_F

from src import dpln

from ..utils import np_feq


class TestDropout(unittest.TestCase):
    def test_dropout_0(self):
        a = np.random.rand(3, 3)
        a_d = dpln.Tensor(a, requires_grad=True)
        b_d = dpln_F.dropout(a_d)
        print(b_d)
        self.assertTrue(b_d.shape == (3, 3))
        g = np.random.rand(3, 3)
        g_d = dpln.Tensor(g)
        b_d.backward(g_d)
        print(a_d.grad)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    unittest.main()
