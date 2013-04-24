#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
from pylstm.layers import ReverseLayer
from pylstm.netbuilder import NetworkBuilder
import numpy as np


class ReverseLayerTest(unittest.TestCase):
    def setUp(self):
        self.size = 3
        self.batches = 1
        self.time_slices = 5
        netb = NetworkBuilder()
        netb.input(self.size) >> ReverseLayer(self.size) >> netb.output
        self.net = netb.build()

    def test_forward_pass(self):
        X = np.arange(self.size * self.batches * self.time_slices).reshape(self.time_slices, self.batches, self.size)
        out = self.net.forward_pass(X)
        self.assertTrue(np.allclose(X[::-1, :, :], out))

    def test_backward_pass(self):
        X = np.arange(self.size * self.batches * self.time_slices).reshape(self.time_slices, self.batches, self.size)
        out = self.net.backward_pass(X)
        self.assertTrue(np.allclose(X[::-1, :, :], -out))



if __name__ == '__main__':
    unittest.main()