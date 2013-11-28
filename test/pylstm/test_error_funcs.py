#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import unittest

from scipy.optimize import approx_fprime
import numpy as np

from pylstm.error_functions import (
    MeanSquaredError, CrossEntropyError, CTC, MultiClassCrossEntropyError)
from pylstm.wrapper import ctcpp_alpha, ctcpp_beta
from pylstm.targets import SequencewiseTargets, LabelingTargets, create_targets_object


class SimpleErrorFuncsTest(unittest.TestCase):
    def setUp(self):
        # error functions under test
        self.error_funcs = [MeanSquaredError,
                            CrossEntropyError,
                            MultiClassCrossEntropyError]

    def test_evaluate_returns_scalar(self):
        Y = np.ones((4, 3, 2))
        T = create_targets_object(np.ones((4, 3, 2)) * 2)
        for err in self.error_funcs:
            e, d = err(Y, T)
            self.assertIsInstance(e, float)

    def test_evaluate_is_batch_normalized(self):
        Y1 = np.ones((4, 1, 2))
        T1 = create_targets_object(np.ones((4, 1, 2)) * 2)
        Y2 = np.ones((4, 10, 2))
        T2 = create_targets_object(np.ones((4, 10, 2)) * 2)
        for err in self.error_funcs:
            e1, d1 = err(Y1, T1)
            e2, d2 = err(Y2, T2)
            self.assertAlmostEqual(e1, e2)

    def test_deriv_shape(self):
        Y = np.ones((4, 3, 2))
        T = create_targets_object(np.ones((4, 3, 2)) * 2)
        for err in self.error_funcs:
            e, d = err(Y, T)
            self.assertEqual(d.shape, Y.shape)

    def test_finite_differences(self):
        Y = np.zeros((4, 3, 2)) + 0.5
        T = create_targets_object(np.ones((4, 3, 2)))
        for err in self.error_funcs:
            def f(X):
                return err(X.reshape(*Y.shape), T)[0]
            delta_approx = approx_fprime(Y.flatten().copy(), f, 1e-7)
            delta_calc = err(Y, T)[1].flatten()
            np.testing.assert_array_almost_equal(delta_approx, delta_calc)


class ClassificationErrorFuncsTest(unittest.TestCase):
    def setUp(self):
        # error functions under test
        self.error_funcs = [MeanSquaredError,
                            CrossEntropyError]

    def test_evaluate_returns_scalar(self):
        Y = np.ones((4, 3, 2))
        T = SequencewiseTargets(np.array([0, 1, 0]), binarize_to=2)
        for err in self.error_funcs:
            e, d = err(Y, T)
            self.assertIsInstance(e, float)

    def test_evaluate_is_batch_normalized(self):
        Y1 = np.ones((4, 1, 2))
        T1 = SequencewiseTargets(np.array([0]), binarize_to=2)
        Y2 = np.ones((4, 10, 2))
        T2 = SequencewiseTargets(np.array([0]*10), binarize_to=2)
        for err in self.error_funcs:
            e1, d1 = err(Y1, T1)
            e2, d2 = err(Y2, T2)
            self.assertAlmostEqual(e1, e2)

    def test_deriv_shape(self):
        Y = np.ones((4, 3, 2))
        T = SequencewiseTargets(np.array([0, 1, 0]), binarize_to=2)
        for err in self.error_funcs:
            e, d = err(Y, T)
            self.assertEqual(d.shape, Y.shape)

    def test_finite_differences(self):
        Y = np.zeros((4, 3, 2)) + 0.5
        T = SequencewiseTargets(np.array([0, 1, 0]), binarize_to=2)
        for err in self.error_funcs:
            print(err)
            def f(X):
                return err(X.reshape(*Y.shape), T)[0]
            delta_approx = approx_fprime(Y.flatten().copy(), f, 1e-7)
            delta_calc = err(Y, T)[1].flatten()
            np.testing.assert_array_almost_equal(delta_approx, delta_calc)


class CTCTest(unittest.TestCase):
    def setUp(self):
        self.Y = np.array([[.1, .7, .2],
                           [.8, .1, .1],
                           [.3, .3, .4],
                           [.7, .1, .2]]).reshape(4, 1, 3)
        self.T = [0, 1]

    def test_alpha_values(self):
        a = ctcpp_alpha(np.log(self.Y), self.T)
        a_expected = np.array(
            [[.1, .08, 0, 0],
             [.7, .08, .048, 0],
             [0, .56, .192, 0],
             [0, .07, .284, .1048],
             [0, 0, .021, .2135]]).T
        self.assertTrue(np.allclose(np.exp(a), a_expected.reshape(4, 1, 5)))

    def test_beta_values(self):
        b = ctcpp_beta(np.log(self.Y), self.T)
        b_expected = np.array(
            [[.096, .06, 0, 0],
             [.441, .48, .2, 0],
             [0, .42, .2, 0],
             [0, .57, .9, 1],
             [0, 0, .7, 1]]).T
        self.assertTrue(np.allclose(np.exp(b), b_expected.reshape(4, 1, 5)))

    def test_alpha_values_duplicate_label(self):
        T = [0, 0]
        a = ctcpp_alpha(np.log(self.Y), T)
        a_expected = np.array(
            [[.1, .08, 0, 0],
             [.7, .08, .048, 0],
             [0, .56, .192, 0],
             [0, 0, .168, .036],
             [0, 0, 0, .1176]]).T
        self.assertTrue(np.allclose(np.exp(a), a_expected.reshape(4, 1, 5)))

    def test_beta_values_duplicate_label(self):
        T = [0, 0]
        b = ctcpp_beta(np.log(self.Y), T)
        b_expected = np.array(
            [[.003, 0, 0, 0],
             [.219, .03, 0, 0],
             [0, .27, .1, 0],
             [0, .45, .8, 1],
             [0, 0, .7, 1]]).T
        self.assertTrue(np.allclose(np.exp(b), b_expected.reshape(4, 1, 5)))

    def test_pxz_equal_for_all_t(self):
        a = ctcpp_alpha(np.log(self.Y), self.T)
        b = ctcpp_beta(np.log(self.Y), self.T)
        pxz = np.exp(a + b).T.sum(0)
        self.assertTrue(np.allclose(pxz, pxz.mean()))

    def test_error_equals_mean_pxz(self):
        a = ctcpp_alpha(np.log(self.Y), self.T)
        b = ctcpp_beta(np.log(self.Y), self.T)
        pzx = np.logaddexp.reduce(a + b, axis=2)
        error_expected = -pzx.mean()
        error_calc, d = CTC(self.Y, LabelingTargets([self.T], binarize_to=3))
        self.assertEqual(error_calc, error_expected)


    def test_finite_diff(self):
        # finite differences testing
        def f(X):
            return CTC(X.reshape(4, 1, 3), LabelingTargets([self.T], binarize_to=3))[0]

        e, d = CTC(self.Y, LabelingTargets([self.T], binarize_to=3))
        print("delta_calc\n", d.reshape(4, 1, 3).T)
        delta_approx = approx_fprime(self.Y.copy().flatten(), f, 1e-5)
        print("delta_approx\n", delta_approx.reshape(4, 1, 3).T)

        self.assertLess(np.sum((delta_approx.reshape(4, 1, 3) - d) ** 2), 1e-4)
