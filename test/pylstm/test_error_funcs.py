#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import unittest
import warnings

from scipy.optimize import approx_fprime
import numpy as np

from pylstm.error_functions import MeanSquaredError, CrossEntropyError, CTC
from pylstm.error_functions import MultiClassCrossEntropyError
from pylstm.error_functions import _ctc_calculate_alphas, _ctc_calculate_betas


class SimpleErrorFuncsTest(unittest.TestCase):
    def setUp(self):
        # error functions under test
        self.error_funcs = [MeanSquaredError,
                            CrossEntropyError,
                            MultiClassCrossEntropyError]

    def test_evaluate_returns_scalar(self):
        Y = np.ones((4, 3, 2))
        T = np.ones((4, 3, 2)) * 2
        for err in self.error_funcs:
            e, d = err(Y, T)
            self.assertIsInstance(e, float)

    def test_evaluate_is_batch_normalized(self):
        Y1 = np.ones((4, 1, 2))
        T1 = np.ones((4, 1, 2)) * 2
        Y2 = np.ones((4, 10, 2))
        T2 = np.ones((4, 10, 2)) * 2
        for err in self.error_funcs:
            e1, d1 = err(Y1, T1)
            e2, d2 = err(Y2, T2)
            self.assertAlmostEqual(e1, e2)

    def test_deriv_shape(self):
        Y = np.ones((4, 3, 2))
        T = np.ones((4, 3, 2)) * 2
        for err in self.error_funcs:
            e, d = err(Y, T)
            self.assertEqual(d.shape, Y.shape)

    def test_finite_differences(self):
        Y = np.zeros((4, 3, 2)) + 0.5
        T = np.ones((4, 3, 2))
        for err in self.error_funcs:
            def f(X):
                return err(X.reshape(*T.shape), T)[0]
            delta_approx = approx_fprime(Y.flatten().copy(), f, 1e-7)
            delta_calc = err(Y, T)[1].flatten()
            np.testing.assert_array_almost_equal(delta_approx, delta_calc)


class CTCTest(unittest.TestCase):

    def setUp(self):
        self.Y = np.array([[.1, .7, .2],
                           [.8, .1, .1],
                           [.3, .3, .4],
                           [.7, .1, .2]])
        self.T = np.array([1, 2])

    def test_alpha_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = _ctc_calculate_alphas(np.log(self.Y), self.T)
            a_expected = np.array(
                [[.1, .08, 0, 0],
                 [.7, .08, .048, 0],
                 [0, .56, .192, 0],
                 [0, .07, .284, .1048],
                 [0, 0, .021, .2135]]).T
            self.assertTrue(np.allclose(np.exp(a), a_expected))

    def test_beta_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = _ctc_calculate_betas(np.log(self.Y), self.T)
            b_expected = np.array(
                [[.096, .06, 0, 0],
                 [.441, .48, .2, 0],
                 [0, .42, .2, 0],
                 [0, .57, .9, 1],
                 [0, 0, .7, 1]]).T
            self.assertTrue(np.allclose(np.exp(b), b_expected))

    def test_alpha_values_duplicate_label(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = np.array([1, 1])
            a = _ctc_calculate_alphas(np.log(self.Y), T)
            a_expected = np.array(
                [[.1, .08, 0, 0],
                 [.7, .08, .048, 0],
                 [0, .56, .192, 0],
                 [0, 0, .168, .036],
                 [0, 0, 0, .1176]]).T
            self.assertTrue(np.allclose(np.exp(a), a_expected))

    def test_beta_values_duplicate_label(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T = np.array([1, 1])
            b = _ctc_calculate_betas(np.log(self.Y), T)
            b_expected = np.array(
                [[.003, 0, 0, 0],
                 [.219, .03, 0, 0],
                 [0, .27, .1, 0],
                 [0, .45, .8, 1],
                 [0, 0, .7, 1]]).T
            self.assertTrue(np.allclose(np.exp(b), b_expected))

    def test_pxz_equal_for_all_t(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = _ctc_calculate_alphas(np.log(self.Y), self.T)
            b = _ctc_calculate_betas(np.log(self.Y), self.T)
            pxz = np.exp(a + b).T.sum(0)
            self.assertTrue(np.allclose(pxz, pxz.mean()))

    def test_finite_diff(self):
        # finite differences testing
        def f(X):
            return CTC(X.reshape(4, 1, 3), [self.T])[0]

        e, d = CTC(self.Y.reshape(4, 1, 3), [self.T])
        print("delta_calc\n", d.reshape(4, 1, 3).T)
        delta_approx = approx_fprime(self.Y.copy().flatten(), f, 1e-5)
        print("delta_approx\n", delta_approx.reshape(4, 1, 3).T)

        self.assertLess(np.sum((delta_approx.reshape(4, 1, 3) - d) ** 2), 1e-4)
