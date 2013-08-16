#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from scipy.optimize import approx_fprime

import numpy as np
import unittest
from pylstm.error_functions import MeanSquaredError, CrossEntropyError, CTC
from pylstm.error_functions import MultiClassCrossEntropyError
from pylstm.error_functions import _ctc_calculate_alphas, _ctc_calculate_betas
import warnings


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
            self.assertEqual(e1, e2)

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

    @unittest.skip
    def test_forward_values_multibatch1(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = np.hstack((self.Y, self.Y))
            T = np.hstack((self.T, self.T))
            a = _ctc_calculate_alphas(np.log(Y), T)
            b = _ctc_calculate_betas(np.log(Y), T)
            a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, .07, .284, .1048], [0, 0, .021, .2135]]).reshape(5, 1, 4)
            b_expected = np.array([[.096, .06, 0, 0], [.441, .48, .2, 0], [0, .42, .2, 0], [0, .57, .9, 1], [0, 0, .7, 1]]).reshape(5, 1, 4)
            self.assertTrue(np.allclose(a[:, 0:1, :].T, a_expected))
            self.assertTrue(np.allclose(b[:, 0:1, :].T, b_expected))
            self.assertTrue(np.allclose(a[:, 1:2, :].T, a_expected))
            self.assertTrue(np.allclose(b[:, 1:2, :].T, b_expected))

    @unittest.skip
    def test_forward_values_multibatch2(self):
        Y = np.hstack((self.Y, self.Y))
        Y[:, 1, :] = 1
        Y /= Y.sum(2).reshape(4, 2, 1)  # normalize to get prob distr
        T = np.hstack((self.T, self.T))
        a, b, d = CTC(Y, T)
        a_expected = np.array([[.1, .08, 0, 0], [.7, .08, .048, 0], [0, .56, .192, 0], [0, .07, .284, .1048], [0, 0, .021, .2135]]).reshape(5, 1, 4)
        b_expected = np.array([[.096, .06, 0, 0], [.441, .48, .2, 0], [0, .42, .2, 0], [0, .57, .9, 1], [0, 0, .7, 1]]).reshape(5, 1, 4)
        self.assertTrue(np.allclose(a[:, 0:1, :].T, a_expected))
        self.assertTrue(np.allclose(b[:, 0:1, :].T, b_expected))

    def test_pxz_equal_for_all_t(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = _ctc_calculate_alphas(np.log(self.Y), self.T)
            b = _ctc_calculate_betas(np.log(self.Y), self.T)
            pxz = np.exp(a + b).T.sum(0)
            self.assertTrue(np.allclose(pxz, pxz.mean()))

    @unittest.skip
    def test_pxz_equal_for_all_t_multibatch(self):
        #Y = np.hstack((self.Y, self.Y))
        Y = np.abs(np.random.randn(4, 2, 3))
        Y /= Y.sum(2).reshape(4, 2, 1)  # normalize to get prob distr
        T = np.hstack((self.T, self.T))
        a, b, d = CTC(Y, T)
        pxz = (a * b).sum(2)

        self.assertTrue(np.allclose(pxz[:,0], pxz[:,0].mean()))
        self.assertTrue(np.allclose(pxz[:,1], pxz[:,1].mean()))

    def test_finite_diff(self):
        # finite differences testing
        def f(X):
            return CTC(X.reshape(4, 1, 3), [self.T])[0]

        e, d = CTC(self.Y.reshape(4, 1, 3), [self.T])
        print("delta_calc\n", d.reshape(4, 1, 3).T)
        delta_approx = approx_fprime(self.Y.copy().flatten(), f, 1e-5)
        print("delta_approx\n", delta_approx.reshape(4, 1, 3).T)

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
        T = np.random.randint(1, labels + 1, (label_seq_length, batches))
        T = [t for t in T.swapaxes(0,1)]

        # finite differences testing
        def f(X):
            return CTC(X.reshape(input_time_size, batches, labels + 1), T)[0]

        delta_approx = approx_fprime(Y.copy().flatten(), f, 1e-5)

        e, d = CTC(Y, T)
        #print("pzx\n", (a*b).sum(2))

        print("approx\n", delta_approx.reshape(input_time_size, batches, labels + 1).T)
        print("calculated\n", d.T)
        print("diff\n", (delta_approx.reshape(input_time_size, batches, labels + 1) - d).T)

        e1, d1 = CTC(Y[:, 0:1, :], T[0:1])
        e2, d2 = CTC(Y[:, 1:2, :], T[1:2])
        d_correct = np.hstack((d1, d2))
        print("delta calculated individually:\n", d_correct.T)
        self.assertLess(np.sum((d.flatten() - delta_approx) ** 2), 1e-4)



