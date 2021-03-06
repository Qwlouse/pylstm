#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import gc
import numpy as np
import unittest

from pylstm.wrapper import Matrix


class MatrixTest(unittest.TestCase):
    def test_construction_with_size(self):
        for i in range(1, 11):
            b = Matrix(i)
            self.assertEqual(len(b), i)
            for j in range(i):
                self.assertEqual(b[j], 0.0)
                
    def test_construction_with_shape(self):
        b = Matrix(2, 3, 4)
        self.assertEqual(len(b), 2 * 3 * 4)
        self.assertEqual(b.get_feature_size(), 4)
        self.assertEqual(b.get_batch_size(), 3)
        self.assertEqual(b.get_time_size(), 2)
        self.assertEqual(b.shape(), (2, 3, 4))

    def test_reshape_view_has_correct_shape(self):
        b = Matrix(12, 2, 1)
        self.assertEqual(b.reshape(4, 3, 2).shape(), (4, 3, 2))
        self.assertEqual(b.reshape(1, 1, 24).shape(), (1, 1, 24))
        self.assertEqual(b.reshape(1, 6, 4).shape(), (1, 6, 4))
        
    def test_reshape_view_with_negative_value_has_correct_shape(self):
        b = Matrix(12, 2, 1)
        self.assertEqual(b.reshape(-1, 3, 2).shape(), (4, 3, 2))
        self.assertEqual(b.reshape(1, 1, -1).shape(), (1, 1, 24))
        self.assertEqual(b.reshape(1, -1, 4).shape(), (1, 6, 4))

    def test_reshape_view_with_multiple_negative_values_raises(self):
        b = Matrix(12, 2, 1)
        self.assertRaises(AssertionError, b.reshape, -1, -1, 1)
        self.assertRaises(AssertionError, b.reshape, -1, 4, -1)
        self.assertRaises(AssertionError, b.reshape, 6, -1, -1)

    def test_reshape_view_with_negative_values_raises_if_indivisible(self):
        b = Matrix(12, 2, 1)
        self.assertRaises(AssertionError, b.reshape, -1, 7, 2)
    
    def test_as_array(self):
        b = Matrix(5, 4, 3)
        a = b.as_array()
        self.assertEqual(a.shape, (5, 4, 3))
        self.assertTrue(np.all(a == 0.))
        
    def test_as_array_is_view(self):
        b = Matrix(5, 4, 3)
        a = b.as_array()
        a[:] = 23.
        self.assertEqual(b[0], 23.)
        self.assertEqual(b[5], 23.)
        self.assertEqual(b[12], 23.)
        
    def test_item_access(self):
        b = Matrix(3)
        b[0] = 5
        b[1] = 7
        b[2] = 10
        self.assertEqual(b[0], 5)
        self.assertEqual(b[1], 7)
        self.assertEqual(b[2], 10)
        
    def test_slicing_view_has_correct_values(self):
        b = Matrix(range(6))
        v = b[1:3]
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 2)
        v = b[4:6]
        self.assertEqual(v[0], 4)
        self.assertEqual(v[1], 5)

    def test_slicing_3dview_has_correct_shapes(self):
        b = Matrix(11, 9, 7)
        self.assertEqual(b[1:3].shape(), (2, 9, 7))
        self.assertEqual(b[3:11].shape(), (8, 9, 7))
        self.assertEqual(b[:3].shape(), (3, 9, 7))
        self.assertEqual(b[5:].shape(), (6, 9, 7))
        self.assertEqual(b[:].shape(), (11, 9, 7))

    def test_construction_with_1d_nparray(self):
        a = np.array([1, 2, 3, 3, 2, 3])
        b = Matrix(a)
        self.assertEqual(len(b), len(a))
        for i in range(len(a)):
            self.assertEqual(b[i], a[i])

    def test_construction_with_3d_nparray(self):
        a = np.array([[[1, 2, 3], [4, 4, 4]], [[1, 2, 3], [4, 4, 4]]])
        b = Matrix(a)
        self.assertEqual(len(b), len(a.flatten()))
        for i in range(len(b)):
            self.assertEqual(b[i], a.flatten()[i])

    def test_slicing_gives_a_view(self):
        b = Matrix(10)
        v = b[1:2]
        self.assertEqual(len(v), 1)
        v = b[0:2]
        self.assertEqual(len(v), 2)
        v = b[0:10]
        self.assertEqual(len(v), 10)
        v = b[4:7]
        self.assertEqual(len(v), 3)
    
    def test_memory_management1(self):
        b = Matrix(10)
        a = b.as_array()
        del b
        gc.collect()
        a[1] = 1
        del a
        gc.collect()
        ## self.assert_no_segfault()  :-)

    def test_memory_management2(self):
        a = np.array([[[1, 2, 3]]])
        b = Matrix(a)
        del a
        gc.collect()
        c = b[1]
        del c
        del b
        gc.collect()
        ## self.assert_no_segfault()  :-)