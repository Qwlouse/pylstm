#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
from pylstm import global_rnd, shuffle_data, build_net, InputLayer, ForwardLayer
from pylstm import Gaussian
from pylstm.randomness import HierarchicalRandomState

import numpy as np


class HierarchicalRandomStateTest(unittest.TestCase):
    def setUp(self):
        self.rnd = HierarchicalRandomState(1)

    def test_constructor_without_arg(self):
        rnd1 = HierarchicalRandomState()
        rnd2 = HierarchicalRandomState()
        self.assertNotEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_constructor_with_seed(self):
        rnd1 = HierarchicalRandomState(2)
        rnd2 = HierarchicalRandomState(2)
        self.assertEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_set_seed(self):
        self.rnd.set_seed(1)
        self.assertEqual(self.rnd.get_seed(), 1)

    def test_randint_randomness(self):
        a = self.rnd.randint(10000)
        b = self.rnd.randint(10000)
        self.assertNotEqual(a, b)

    def test_seeded_randint_deterministic(self):
        self.rnd.set_seed(1)
        a = self.rnd.randint(10000)
        self.rnd.set_seed(1)
        b = self.rnd.randint(10000)
        self.assertEqual(a, b)

    def test_get_new_random_state_randomness(self):
        rnd1 = self.rnd.get_new_random_state()
        rnd2 = self.rnd.get_new_random_state()
        self.assertNotEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_seeded_get_new_random_state_deterministic(self):
        self.rnd.set_seed(1)
        rnd1 = self.rnd.get_new_random_state()
        self.rnd.set_seed(1)
        rnd2 = self.rnd.get_new_random_state()
        self.assertEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_get_item_randomness(self):
        rnd1 = self.rnd['A']
        rnd2 = self.rnd['A']
        self.assertNotEqual(rnd1.randint(1000), rnd2.randint(1000))

    def test_seeded_get_item_deterministic(self):
        self.rnd.set_seed(1)
        rnd1 = self.rnd['A']
        self.rnd.set_seed(1)
        rnd2 = self.rnd['A']
        self.assertEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_seeded_get_item_deterministic2(self):
        self.rnd.set_seed(1)
        rnd1 = self.rnd['A']
        rnd2 = self.rnd['A']
        self.assertEqual(rnd1, rnd2)

    def test_get_item_independent_of_previous_usage(self):
        self.rnd.set_seed(1)
        rnd1 = self.rnd['A']
        self.rnd.set_seed(1)
        self.rnd.randint(1000)
        rnd2 = self.rnd['A']
        self.assertNotEqual(rnd1, rnd2)
        self.assertEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_get_item_different_names(self):
        rnd1 = self.rnd['A']
        rnd2 = self.rnd['B']
        self.assertNotEqual(rnd1, rnd2)


class GlobalRndTest(unittest.TestCase):
    def setUp(self):
        global_rnd.set_seed(1)

    def test_global_rnd_randomness(self):
        self.assertNotEqual(global_rnd.randint(1000), global_rnd.randint(1000))

    def test_seeded_global_rnd_deterministic(self):
        global_rnd.set_seed(1)
        a = global_rnd.randint(1000)
        global_rnd.set_seed(1)
        b = global_rnd.randint(1000)
        self.assertEqual(a, b)

    def test_shuffle_is_random(self):
        X = np.arange(10).reshape(1, -1, 1)
        T = np.arange(10).reshape(1, -1, 1)
        M = None
        global_rnd.set_seed(1)
        _, _, _, s1 = shuffle_data(X, T, M)
        _, _, _, s2 = shuffle_data(X, T, M)
        self.assertFalse(np.all(s1 == s2))

    def test_shuffle_depends_on_global_seed(self):
        X = np.arange(10).reshape(1, -1, 1)
        T = np.arange(10).reshape(1, -1, 1)
        M = None
        global_rnd.set_seed(1)
        _, _, _, s1 = shuffle_data(X, T, M)
        global_rnd.set_seed(1)
        _, _, _, s2 = shuffle_data(X, T, M)
        np.testing.assert_array_equal(s1.flat, s2.flat)

    def test_shuffle_seed_overwrites_global_seed(self):
        X = np.arange(10).reshape(1, -1, 1)
        T = np.arange(10).reshape(1, -1, 1)
        M = None
        global_rnd.set_seed(1)
        _, _, _, s1 = shuffle_data(X, T, M, seed=1)
        global_rnd.set_seed(1)
        _, _, _, s2 = shuffle_data(X, T, M, seed=2)
        self.assertFalse(np.all(s1 == s2))

    def test_initialize_is_random(self):
        net1 = build_net(InputLayer(1) >> ForwardLayer(3))
        net1.initialize(Gaussian())
        net2 = build_net(InputLayer(1) >> ForwardLayer(3))
        net2.initialize(Gaussian())
        self.assertFalse(np.allclose(net1.param_buffer, net2.param_buffer))

    def test_initialize_depends_on_global_rnd(self):
        global_rnd.set_seed(1)
        net1 = build_net(InputLayer(1) >> ForwardLayer(3))
        net1.initialize(Gaussian())
        global_rnd.set_seed(1)
        net2 = build_net(InputLayer(1) >> ForwardLayer(3))
        net2.initialize(Gaussian())
        np.testing.assert_allclose(net1.param_buffer, net2.param_buffer)




