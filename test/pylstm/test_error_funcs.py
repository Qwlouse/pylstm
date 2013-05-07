#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from scipy.optimize import approx_fprime

import numpy as np
import unittest
from pylstm.error_functions import CTC, MeanSquaredError, CrossEntropyError
from pylstm.error_functions import MultiClassCrossEntropyError


class SimpleErrorFuncsTest(unittest.TestCase):
    def setUp(self):
        # error functions under test
        self.error_funcs = [MeanSquaredError(),
                            CrossEntropyError(),
                            MultiClassCrossEntropyError()]

    def test_evaluate_returns_scalar(self):
        Y = np.ones((4, 3, 2))
        T = np.ones((4, 3, 2)) * 2
        for err in self.error_funcs:
            e = err.evaluate(Y, T)
            self.assertIsInstance(e, float)

    def test_evaluate_is_batch_normalized(self):
        Y1 = np.ones((4, 1, 2))
        T1 = np.ones((4, 1, 2)) * 2
        Y2 = np.ones((4, 10, 2))
        T2 = np.ones((4, 10, 2)) * 2
        for err in self.error_funcs:
            e1 = err.evaluate(Y1, T1)
            e2 = err.evaluate(Y2, T2)
            self.assertEqual(e1, e2)

    def test_deriv_shape(self):
        Y = np.ones((4, 3, 2))
        T = np.ones((4, 3, 2)) * 2
        for err in self.error_funcs:
            d = err.deriv(Y, T)
            self.assertEqual(d.shape, Y.shape)

    def test_finite_differences(self):
        Y = np.zeros((4, 3, 2)) + 0.5
        T = np.ones((4, 3, 2))
        for err in self.error_funcs:
            def f(X):
                return err.evaluate(X.reshape(*T.shape), T)
            delta_approx = approx_fprime(Y.flatten().copy(), f, 1e-7)
            delta_calc = err.deriv(Y, T).flatten()
            np.testing.assert_array_almost_equal(delta_approx, delta_calc)


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

    def test_forward_values_duplicate_label(self):
        T = np.array([1, 1]).reshape(-1, 1, 1)
        a, b, d = self.ctc(self.Y, T)
        a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, 0, .168, .036], [0, 0, 0, .1176]]).reshape(5, 1, 4).T
        b_expected = np.array([[.003, 0, 0, 0], [.219, .03, 0, 0], [0, .27, .1, 0], [0, .45, .8, 1], [0, 0, .7, 1]]).reshape(5, 1, 4).T
        print("calculated\n:", b.T)
        print("expected\n:", b_expected.T)
        self.assertTrue(np.allclose(a, a_expected))
        self.assertTrue(np.allclose(b, b_expected))

    def test_forward_values_multibatch1(self):
        Y = np.hstack((self.Y, self.Y))
        T = np.hstack((self.T, self.T))
        a, b, d = self.ctc(Y, T)
        a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, .07, .284, .1048], [0, 0, .021, .2135]]).reshape(5, 1, 4)
        b_expected = np.array([[.096, .06, 0, 0], [.441, .48, .2, 0], [0, .42, .2, 0], [0, .57, .9, 1], [0, 0, .7, 1]]).reshape(5, 1, 4)
        self.assertTrue(np.allclose(a[:, 0:1, :].T, a_expected))
        self.assertTrue(np.allclose(b[:, 0:1, :].T, b_expected))
        self.assertTrue(np.allclose(a[:, 1:2, :].T, a_expected))
        self.assertTrue(np.allclose(b[:, 1:2, :].T, b_expected))

    def test_forward_values_multibatch2(self):
        Y = np.hstack((self.Y, self.Y))
        Y[:, 1, :] = 1
        Y /= Y.sum(2).reshape(4, 2, 1)  # normalize to get prob distr
        T = np.hstack((self.T, self.T))
        a, b, d = self.ctc(Y, T)
        a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, .07, .284, .1048], [0, 0, .021, .2135]]).reshape(5, 1, 4)
        b_expected = np.array([[.096, .06, 0, 0], [.441, .48, .2, 0], [0, .42, .2, 0], [0, .57, .9, 1], [0, 0, .7, 1]]).reshape(5, 1, 4)
        self.assertTrue(np.allclose(a[:, 0:1, :].T, a_expected))
        self.assertTrue(np.allclose(b[:, 0:1, :].T, b_expected))

    def test_pxz_equal_for_all_t(self):
        a, b, d = self.ctc(self.Y, self.T)
        pxz = (a * b).T.sum(0)
        self.assertTrue(np.allclose(pxz, pxz.mean()))

    def test_pxz_equal_for_all_t_multibatch(self):
        #Y = np.hstack((self.Y, self.Y))
        Y = np.abs(np.random.randn(4, 2, 3))
        Y /= Y.sum(2).reshape(4, 2, 1)  # normalize to get prob distr
        T = np.hstack((self.T, self.T))
        a, b, d = self.ctc(Y, T)
        pxz = (a * b).sum(2)

        self.assertTrue(np.allclose(pxz[:,0], pxz[:,0].mean()))
        self.assertTrue(np.allclose(pxz[:,1], pxz[:,1].mean()))

    def test_finite_diff(self):
        # finite differences testing
        def f(X):
            a, b, d = self.ctc(X.reshape(4, 1, 3), self.T)
            return -np.log((a * b).T.sum(0).mean())

        a, b, d = self.ctc(self.Y.reshape(4, 1, 3), self.T)
        delta_approx = approx_fprime(self.Y.copy().flatten(), f, 1e-5)
        print("delta_approx\n", delta_approx.reshape(4, 3).T)
        self.assertLess(np.sum((delta_approx.reshape(4, 1, 3) - d) ** 2), 1e-4)

    def test_finite_diff_multibatch(self):
        batches = 7
        input_time_size = 9
        labels = 2
        label_seq_length = 4
        Y = np.abs(np.random.randn(input_time_size, batches, labels + 1))
        Y /= Y.sum(2).reshape(input_time_size, batches, 1)  # normalize to get prob distr
        # T = np.vstack([np.arange(1, labels + 1)] * batches).T  # no repeated numbers for now
        # T = T.reshape(labels, batches, 1)
        T = np.random.randint(1, labels + 1, (label_seq_length, batches, 1))

        # finite differences testing
        # TODO: This is wrong somehow:
        def f(X):
            a, b, d = self.ctc(X.reshape(input_time_size, batches, labels + 1), T)
            #return -(np.log((a * b).sum(2))).sum(1).mean()  # probably this line
            return -(np.log((a * b).sum(2))).sum(1).mean()  # probably this line

        delta_approx = approx_fprime(Y.copy().flatten(), f, 1e-5)

        a, b, d = self.ctc(Y, T)
        #print("pzx\n", (a*b).sum(2))

        print("approx\n", delta_approx.reshape(input_time_size, batches, labels + 1).T)
        print("calculated\n", d.T)
        print("diff\n", (delta_approx.reshape(input_time_size, batches, labels + 1) - d).T)

        a1, b1, d1 = self.ctc(Y[:, 0:1, :], T[:, 0:1, :])
        a2, b2, d2 = self.ctc(Y[:, 1:2, :], T[:, 1:2, :])
        d_correct = np.hstack((d1, d2))
        print("delta calculated individually:\n", d_correct.T)
        a_corr = np.hstack((a1, a2))
        print("alpha calc\n", a.T)
        print("alpha calc individually\n", a_corr.T)

        b_corr = np.hstack((b1, b2))
        print("beta calc\n", b.T)
        print("beta calc individually\n", b_corr.T)

        self.assertLess(np.sum((d.flatten() - delta_approx) ** 2), 1e-4)



