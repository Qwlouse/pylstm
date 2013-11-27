#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest

import numpy as np

from pylstm.structure import build_net, ReverseLayer, InputLayer


class ReverseLayerTest(unittest.TestCase):
    def setUp(self):
        self.size = 3
        self.batches = 1
        self.time_slices = 5

        self.net = build_net(InputLayer(self.size) >> ReverseLayer())

    def test_forward_pass(self):
        X = np.arange(self.size * self.batches * self.time_slices).reshape(
            self.time_slices, self.batches, self.size)
        out = self.net.forward_pass(X)
        print(X[::-1, :, :])
        print(out)
        self.assertTrue(np.allclose(X[::-1, :, :], out))

    def test_backward_pass(self):
        X = np.arange(self.size * self.batches * self.time_slices).reshape(
            self.time_slices, self.batches, self.size)
        self.net.forward_pass(X)
        out = self.net.pure_backpass(X.copy())
        print(X[::-1, :, :])
        print(out)
        np.testing.assert_array_almost_equal(X[::-1, :, :], out)


if __name__ == '__main__':
    unittest.main()