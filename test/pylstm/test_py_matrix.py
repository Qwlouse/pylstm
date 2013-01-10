#!/usr/bin/python
# coding=utf-8
import unittest
import numpy as np
import pylstm.py_matrix as pm


class MatrixCPUTest(unittest.TestCase):
    def test_matrix_from_lists_has_correct_shape(self):
        a = [[[1, 2], [3, 4], [5, 6]], [[2, 3], [4, 5], [6, 7]],
             [[3, 4], [5, 6], [7, 8]], [[4, 5], [6, 7], [8, 9]]]
        npa = np.array(a)
        self.assertEqual(npa.shape, (4, 3, 2))
        m = pm.MatrixCPU(a)
        self.assertEqual(m.get_feature_count(), 2)
        self.assertEqual(m.get_batch_count(), 3)
        self.assertEqual(m.get_slice_count(), 4)

