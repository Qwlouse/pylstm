#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
from pylstm.randomness import HierarchicalRandomState
from pylstm import global_rnd, set_global_seed


class HierarchicalRandomStateTest(unittest.TestCase):
    def setUp(self):
        self.rnd = HierarchicalRandomState(1)

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
        self.assertNotEqual(rnd1.get_seed(), rnd2.get_seed())

    def test_seeded_get_item_deterministic(self):
        self.rnd.set_seed(1)
        rnd1 = self.rnd['A']
        self.rnd.set_seed(1)
        rnd2 = self.rnd['A']
        self.assertNotEqual(rnd1.get_seed(), rnd2.get_seed())


