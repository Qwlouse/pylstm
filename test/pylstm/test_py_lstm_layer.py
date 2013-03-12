#!/usr/bin/python
# coding=utf-8
import unittest
import pylstm.wrapper as pw

class LstmLayerTest(unittest.TestCase):
    def test_in_out_sizes_match(self):
        sizes = [(1, 1), (1, 9), (9, 1), (9, 9), (3, 7)]
        for n, m in sizes:
            l = pw.create_layer("LstmLayer", n, m)
            self.assertEqual(l.in_size, n)
            self.assertEqual(l.out_size, m)

    def test_param_size(self):
        sizes = [(1, 1), (1, 9), (9, 1), (9, 9), (3, 7)]
        for n, m in sizes:
            l = pw.create_layer("LstmLayer", n, m)
            param_size = l.get_param_size()
            self.assertGreaterEqual(param_size, n * m)

    def test_internal_size(self):
        sizes = [(1, 1), (1, 9), (9, 1), (9, 9), (3, 7)]
        for n, m in sizes:
            l = pw.create_layer("LstmLayer", n, m)
            internal_size = l.get_internal_state_size()
            self.assertGreaterEqual(internal_size, m)

    def test_create_param_view(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        wm = pw.Buffer(l.get_param_size())
        W = l.create_param_view(wm)
        self.assertIsNotNone(W)

    def test_create_internal_view(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        im = pw.Buffer(l.get_internal_state_size())
        I = l.create_internal_view(im)
        self.assertIsNotNone(I)

    def test_create_input_view_with_single_sample(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        im = pw.Buffer(l.in_size)
        I = l.create_input_view(im)
        self.assertIsNotNone(I)
        self.assertEqual(len(I), 3)
        self.assertEqual(I.shape(), (1, 1, 3))

    def test_create_input_view_with_3d_buffer(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        t = 9
        b = 5
        im = pw.Buffer(t, b, l.in_size)
        I = l.create_input_view(im, t, b)
        self.assertIsNotNone(I)
        self.assertEqual(len(I), 3 * t * b)
        self.assertEqual(I.shape(), (t, b, 3))

    def test_create_input_view_with_1d_buffer(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        t = 9
        b = 5
        im = pw.Buffer( l.get_input_buffer_size(t, b))
        I = l.create_input_view(im, t, b)
        self.assertIsNotNone(I)
        self.assertEqual(len(I), 3 * t * b)
        self.assertEqual(I.shape(), (t, b, 3))

    def test_create_output_view_with_single_sample(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        im = pw.Buffer(l.out_size)
        I = l.create_output_view(im)
        self.assertIsNotNone(I)
        self.assertEqual(len(I), 7)
        self.assertEqual(I.shape(), (1, 1, 7))

    def test_create_output_view_with_3d_buffer(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        t = 9
        b = 5
        im = pw.Buffer(t, b, l.out_size)
        I = l.create_output_view(im, t, b)
        self.assertIsNotNone(I)
        self.assertEqual(len(I), 7 * t * b)
        self.assertEqual(I.shape(), (t, b, 7))

    def test_create_output_view_with_1d_buffer(self):
        l = pw.create_layer("LstmLayer", 3, 7)
        t = 9
        b = 5
        im = pw.Buffer( l.get_output_buffer_size(t, b))
        I = l.create_output_view(im, t, b)
        self.assertIsNotNone(I)
        self.assertEqual(len(I), 7 * t * b)
        self.assertEqual(I.shape(), (t, b, 7))

    def test_forward_pass(self):
        t = 1  # time
        b = 1  # batches
        n = 5  # input size
        m = 3  # output size

        l = pw.create_layer("LstmLayer", n, m)
        X = pw.Buffer(t, b, n)
        Y = pw.Buffer(t, b, m)
        wm = pw.Buffer(1, 1, l.get_param_size())
        W = l.create_param_view(wm)
        im = pw.Buffer(t, b, l.get_internal_state_size(t, b))
        I = l.create_internal_view(im)
        l.forward(W, I, X, Y)
