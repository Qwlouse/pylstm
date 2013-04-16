#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from scipy.optimize import approx_fprime

import numpy as np
import unittest
from pylstm.error_functions import CTC


class CTCTest(unittest.TestCase):

    def setUp(self):
        self.ctc = CTC()
        self.Y = np.array([[.1, .7, .2], [.8, .1, .1], [.3, .3, .4], [.7, .1, .2]]).reshape(4, 1, 3)
        self.T = np.array([1, 2]).reshape(-1, 1, 1)

    def test_forward_values(self):
        a, b, d = self.ctc(self.Y, self.T)
        a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, .07, .284, .1048], [0, 0, .021, .2135]]).reshape(5, 1, 4).T
        b_expected = np.array([[.096, .06, 0, 0], [.441, .48, .2, 0], [0, .42, .2, 0], [0, .57, .9, 1], [0, 0, .7, 1]]).reshape(5, 1, 4).T
        self.assertTrue(np.allclose(a, a_expected))
        self.assertTrue(np.allclose(b, b_expected))

    def test_pxz_equal_for_all_t(self):
        a, b, d = self.ctc(self.Y, self.T)
        pxz = (a * b).T.sum(0)
        self.assertTrue(np.allclose(pxz, pxz.mean()))

    def test_finite_diff(self):
        # finite differences testing
        def f(X):
            a, b, d = self.ctc(X.reshape(4, 1, 3), self.T)
            return -np.log((a * b).T.sum(0).mean())

        delta_approx = approx_fprime(self.Y.copy().flatten(), f, 1e-5)
        print("delta_approx\n", delta_approx.reshape(4, 3).T)
        self.assertLess(np.sum(delta_approx ** 2), 1e-4)

    def test_finite_diff_multibatch(self):
        batches = 5
        input_time_size = 6
        labels = 2
        label_seq_length = 2
        Y = np.abs(np.random.randn(input_time_size, batches, labels + 1))
        Y /= Y.sum(2).reshape(input_time_size, batches, 1)  # normalize to get prob distr
        T = np.random.randint(1, labels + 1, (label_seq_length, batches, 1))

        # finite differences testing
        def f(X):
            a, b, d = self.ctc(X.reshape(input_time_size, batches, labels + 1), T)
            return -np.log((a * b).T.sum(0).mean())

        delta_approx = approx_fprime(Y.copy().flatten(), f, 1e-5)
        print("delta_approx\n", delta_approx.reshape(input_time_size, batches, labels + 1).T)
        self.assertLess(np.sum(delta_approx ** 2), 1e-4)