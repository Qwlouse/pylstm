#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import unittest
import numpy as np
from numpy.testing import *
from pylstm import FramewiseTargets
from pylstm.datasets.preprocessing import (get_means, subtract_means, get_stds,
                                           divide_by_stds, center_dataset,
                                           scale_std_of_dataset)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(24, dtype=np.float64).reshape((3, 4, 2))
        self.M = np.ones((3, 4, 1))
        self.M[2, 0, 0] = 0
        self.M[1:, 1, 0] = 0
        self.M[2, 3, 0] = 0
        self.cm = np.array([False, True])

    #################### get_means ####################

    def test_get_means(self):
        means = get_means(self.X)
        assert_allclose(means, [11., 12.])

    def test_get_means_masked(self):
        means = get_means(self.X, self.M)
        assert_allclose(means, [8.25, 9.25])

    def test_get_means_channel_masked(self):
        means = get_means(self.X, channel_mask=self.cm)
        assert_allclose(means, [12])

    def test_get_means_masked_and_channel_masked(self):
        means = get_means(self.X, mask=self.M,
                          channel_mask=self.cm)
        assert_allclose(means, [9.25])

    #################### subtract_means ####################

    def test_subtract_means(self):
        subtract_means(self.X, [11, 12])
        means = get_means(self.X)
        assert_allclose(means, [0., 0.])

    def test_subtract_means_masked(self):
        subtract_means(self.X, [8.25, 9.25], mask=self.M)
        means = get_means(self.X, self.M)
        assert_allclose(means, [0., 0.])
        self.assertEqual(self.X[2, 0, 0], 16.)
        self.assertEqual(self.X[2, 0, 1], 17.)

    def test_subtract_means_channel_masked(self):
        subtract_means(self.X, [12], channel_mask=self.cm)
        means = get_means(self.X, channel_mask=self.cm)
        assert_allclose(means, [0.])
        means = get_means(self.X)
        assert_allclose(means, [11., 0.])

    def test_subtract_means_masked_and_channel_masked(self):
        subtract_means(self.X, [9.25], mask=self.M,
                       channel_mask=self.cm)
        means = get_means(self.X, mask=self.M, channel_mask=self.cm)
        assert_allclose(means, [0.])
        means = get_means(self.X, mask=self.M)
        assert_allclose(means, [8.25, 0.])

    #################### get_stds ####################

    def test_get_stds(self):
        stds = get_stds(self.X)
        assert_allclose(stds, [6.90410506, 6.90410506])

    def test_get_stds_masked(self):
        stds = get_stds(self.X, self.M)
        assert_allclose(stds, [6.27992834, 6.27992834])

    def test_get_stds_channel_masked(self):
        stds = get_stds(self.X, channel_mask=np.array([False, True]))
        assert_allclose(stds, [6.90410506])

    def test_get_stds_masked_and_channel_masked(self):
        stds = get_stds(self.X, mask=self.M,
                        channel_mask=np.array([False, True]))
        assert_allclose(stds, [6.27992834])

    #################### divide_by_stds ####################

    def test_divide_by_stds(self):
        divide_by_stds(self.X, [6.90410506, 6.90410506])
        stds = get_stds(self.X)
        assert_allclose(stds, [1.0, 1.0])

    def test_divide_by_stds_masked(self):
        divide_by_stds(self.X, [6.27992834, 6.27992834], mask=self.M)
        stds = get_stds(self.X, self.M)
        assert_allclose(stds, [1.0, 1.0])

    def test_divide_by_stds_channel_masked(self):
        divide_by_stds(self.X, [6.90410506], channel_mask=self.cm)
        stds = get_stds(self.X, channel_mask=self.cm)
        assert_allclose(stds, [1.0])
        stds = get_stds(self.X)
        assert_allclose(stds, [6.90410506, 1.0])

    def test_divide_by_stds_masked_and_channel_masked(self):
        divide_by_stds(self.X, [6.27992834], mask=self.M,
                       channel_mask=self.cm)
        stds = get_stds(self.X, mask=self.M, channel_mask=self.cm)
        assert_allclose(stds, [1.])
        stds = get_stds(self.X, mask=self.M)
        assert_allclose(stds, [6.27992834, 1.0])

    #################### center_dataset ####################

    def test_center_dataset(self):
        ds = {
            'training': (self.X, FramewiseTargets(self.X)),
            'validation': (self.X + 1, FramewiseTargets(self.X)),
            'test': (self.X * 2, FramewiseTargets(self.X))
        }

        assert_allclose(get_means(ds['training'][0]), [11., 12.])
        assert_allclose(get_means(ds['validation'][0]), [12., 13.])
        assert_allclose(get_means(ds['test'][0]), [22., 24.])

        center_dataset(ds)

        assert_allclose(get_means(ds['training'][0]), [0., 0.], atol=1e-6)
        assert_allclose(get_means(ds['validation'][0]), [1., 1.], atol=1e-6)
        assert_allclose(get_means(ds['test'][0]), [11., 12.], atol=1e-6)

    def test_center_dataset_masked(self):
        ds = {
            'training': (self.X, FramewiseTargets(self.X, self.M)),
            'validation': (self.X + 1, FramewiseTargets(self.X, self.M)),
            'test': (self.X * 2, FramewiseTargets(self.X, self.M))
        }
        assert_allclose(get_means(ds['training'][0], self.M), [8.25, 9.25])
        assert_allclose(get_means(ds['validation'][0], self.M), [9.25, 10.25])
        assert_allclose(get_means(ds['test'][0], self.M), [16.5, 18.5])

        center_dataset(ds)

        assert_allclose(get_means(ds['training'][0], self.M), [0., 0.])
        assert_allclose(get_means(ds['validation'][0], self.M), [1., 1.])
        assert_allclose(get_means(ds['test'][0], self.M), [8.25, 9.25])

    #################### center_dataset ####################

    def test_scale_std_of_dataset(self):
        ds = {
            'training': (self.X, FramewiseTargets(self.X)),
            'validation': (self.X + 1, FramewiseTargets(self.X)),
            'test': (self.X * 2, FramewiseTargets(self.X))
        }

        assert_allclose(get_stds(ds['training'][0]), [6.90410506, 6.90410506])
        assert_allclose(get_stds(ds['validation'][0]), [6.90410506, 6.90410506])
        assert_allclose(get_stds(ds['test'][0]), [13.80821012,  13.80821012])

        scale_std_of_dataset(ds)

        assert_allclose(get_stds(ds['training'][0]), [1., 1.], atol=1e-6)
        assert_allclose(get_stds(ds['validation'][0]), [1., 1.], atol=1e-6)
        assert_allclose(get_stds(ds['test'][0]), [2., 2.], atol=1e-6)

    def test_scale_std_of_dataset_masked(self):
        ds = {
            'training': (self.X, FramewiseTargets(self.X, self.M)),
            'validation': (self.X + 1, FramewiseTargets(self.X, self.M)),
            'test': (self.X * 2, FramewiseTargets(self.X, self.M))
        }
        assert_allclose(get_stds(ds['training'][0], self.M),
                        [6.27992834, 6.27992834])
        assert_allclose(get_stds(ds['validation'][0], self.M),
                        [6.27992834, 6.27992834])
        assert_allclose(get_stds(ds['test'][0], self.M),
                        [12.55985668,  12.55985668])

        scale_std_of_dataset(ds)

        assert_allclose(get_stds(ds['training'][0], self.M), [1., 1.])
        assert_allclose(get_stds(ds['validation'][0], self.M), [1., 1.])
        assert_allclose(get_stds(ds['test'][0], self.M), [2., 2.])
