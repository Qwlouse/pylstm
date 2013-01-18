#!/usr/bin/python
# coding=utf-8
import unittest
import numpy as np
import pylstm.pylstm_wrapper as pw


class LstmLayerTest(unittest.TestCase):
    def test_in_out_sizes_match(self):
        sizes = [(1, 1), (1, 9), (9, 1), (9, 9), (3, 7)]
        for n, m in sizes:
            l = pw.LstmLayer(n, m)
            self.assertEqual(l.get_input_size(), n)
            self.assertEqual(l.get_output_size(), m)

    def test_param_size(self):
        sizes = [(1, 1), (1, 9), (9, 1), (9, 9), (3, 7)]
        for n, m in sizes:
            l = pw.LstmLayer(n, m)
            param_size = l.get_param_size()
            self.assertGreaterEqual(param_size, n*m)

    def test_internal_size(self):
        sizes = [(1, 1), (1, 9), (9, 1), (9, 9), (3, 7)]
        for n, m in sizes:
            l = pw.LstmLayer(n, m)
            internal_size = l.get_internal_state_size()
            self.assertGreaterEqual(internal_size, n*m)

    def test_create_param_view(self):
        l = pw.LstmLayer(3, 7)
        wm = pw.MatrixCPU(1, 1, l.get_param_size())
        W = l.create_param_view(wm)
        self.assertIsNotNone(W)

    def test_create_internal_view(self):
        l = pw.LstmLayer(3, 7)
        im = pw.MatrixCPU(1, 1, l.get_internal_state_size())
        I = l.create_internal_view(im)
        self.assertIsNotNone(I)

    def test_forward_pass(self):
        t = 1 # time
        b = 1 # batches
        n = 5 # input size
        m = 3 # output size

        l = pw.LstmLayer(n, m)
        X = pw.MatrixCPU(t, b, n)
        Y = pw.MatrixCPU(t, b, m)
        wm = pw.MatrixCPU(1, 1, l.get_param_size())
        W = l.create_param_view(wm)
        im = pw.MatrixCPU(t, b, l.get_internal_state_size(t, b))
        I = l.create_internal_view(im)
        l.forward(W, I, X, Y)

