#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import unittest
import numpy as np
from pylstm.pylstm_wrapper import BufferView as Buffer


class BufferTest(unittest.TestCase):
    def test_constrution_with_size(self):
        for i in range(1, 11):
            b = Buffer(i)
            self.assertEqual(len(b), i)
            for j in range(i):
                self.assertEqual(b[j], 0.0)

    def test_construction_with_1d_nparray(self):
        a = np.array([1, 2, 3, 3, 2, 3])
        b = Buffer(a)
        self.assertEqual(len(b), len(a))
        for i in range(len(a)):
            self.assertEqual(b[i], a[i])

    def test_construction_with_3d_nparray(self):
        a = np.array([[[1,2,3],[4,4,4]],[[1,2,3],[4,4,4]]])
        b = Buffer(a)
        self.assertEqual(len(b), len(a.flatten()))
        for i in range(len(b)):
            self.assertEqual(b[i], a.flatten()[i])

    def test_slicing_gives_a_view(self):
        b = Buffer(10)
        v = b[1:2]
        self.assertEqual(len(v), 1)
        v = b[0:2]
        self.assertEqual(len(v), 2)
        v = b[0:10]
        self.assertEqual(len(v), 10)
        v = b[4:7]
        self.assertEqual(len(v), 3)

    def test_slicing_view_has_correct_values(self):
        a = np.arange(6)
        b = Buffer(a)
        v = b[1:3]
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        v = b[4:6]
        self.assertEqual(v[0], 4)
        self.assertEqual(v[1], 5)
