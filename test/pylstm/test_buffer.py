#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import unittest
import numpy as np
from pylstm.wrapper import BufferView


class BufferTest(unittest.TestCase):
    def test_constrution_with_size(self):
        for i in range(1, 11):
            b = BufferView(i)
            self.assertEqual(len(b), i)
            for j in range(i):
                self.assertEqual(b[j], 0.0)

    def test_construction_with_1d_nparray(self):
        a = np.array([1, 2, 3, 3, 2, 3])
        b = BufferView(a)
        self.assertEqual(len(b), len(a))
        for i in range(len(a)):
            self.assertEqual(b[i], a[i])

    def test_construction_with_3d_nparray(self):
        a = np.array([[[1,2,3],[4,4,4]],[[1,2,3],[4,4,4]]])
        b = BufferView(a)
        self.assertEqual(len(b), len(a.flatten()))
        for i in range(len(b)):
            self.assertEqual(b[i], a.flatten()[i])

    def test_slicing_gives_a_view(self):
        b = BufferView(10)
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
        b = BufferView(a)
        v = b[1:3]
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        v = b[4:6]
        self.assertEqual(v[0], 4)
        self.assertEqual(v[1], 5)

    def test_slicing_3dview_has_correct_shapes(self):
        a = np.zeros((11, 9, 7))
        b = BufferView(a)
        self.assertEqual(b[1:3].shape(), (2, 9, 7))
        self.assertEqual(b[3:11].shape(), (8, 9, 7))
        self.assertEqual(b[:3].shape(), (3, 9, 7))
        self.assertEqual(b[5:].shape(), (6, 9, 7))
        self.assertEqual(b[:].shape(), (11, 9, 7))

    def test_reshape_view_has_correct_shape(self):
        a = np.zeros((12, 2, 1))
        b = BufferView(a)
        self.assertEqual(b.reshape(4, 3, 2).shape(), (4, 3, 2))
        self.assertEqual(b.reshape(1, 1, 24).shape(), (1, 1, 24))
        self.assertEqual(b.reshape(1, 6, 4).shape(), (1, 6, 4))

    def test_reshape_view_with_negative_value_has_correct_shape(self):
        a = np.zeros((12, 2, 1))
        b = BufferView(a)
        self.assertEqual(b.reshape(-1, 3, 2).shape(), (4, 3, 2))
        self.assertEqual(b.reshape(1, 1, -1).shape(), (1, 1, 24))
        self.assertEqual(b.reshape(1, -1, 4).shape(), (1, 6, 4))

    def test_reshape_view_with_multiple_negative_values_raises(self):
        a = np.zeros((12, 2, 1))
        b = BufferView(a)
        self.assertRaises(AssertionError, b.reshape, -1, -1, 1)
        self.assertRaises(AssertionError, b.reshape, -1, 4, -1)
        self.assertRaises(AssertionError, b.reshape, 6, -1, -1)

    def test_reshape_view_with_negative_values_raises_if_indivisible(self):
        a = np.zeros((12, 2, 1))
        b = BufferView(a)
        self.assertRaises(AssertionError, b.reshape, -1, 7, 2)