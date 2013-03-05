#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
import numpy as np
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import NpFwdLayer
from pylstm.wrapper import Buffer


class NetworkTests(unittest.TestCase):
    def setUp(self):
        netb = NetworkBuilder()
        netb.input(5) >> NpFwdLayer(3) >> netb.output
        self.net = netb.build()
        self.net.set_param_buffer(np.random.randn(self.net.get_param_size()))
        self.X = np.random.randn(2, 7, self.net.get_input_size())

    def test_lstm_forward_pass_insensitive_to_internal_state(self):
        out1 = self.net.forward_pass(self.X).as_array().copy()
        self.net.intern_manager.initialize_buffer(Buffer(np.random.randn(
            self.net.intern_manager.calculate_size())))
        out2 = self.net.forward_pass(self.X).as_array().copy()
        self.assertTrue(np.allclose(out1, out2))

    def test_lstm_backward_pass_insensitive_to_internal_deltas(self):
        self.net.clear_internal_state()
        out1 = self.net.forward_pass(self.X).as_array().copy()
        deltas1 = self.net.backward_pass(out1).as_array().copy()
        self.net.intern_manager.initialize_buffer(Buffer(np.random.randn(
            self.net.intern_manager.calculate_size())))
        self.net.delta_manager.initialize_buffer(Buffer(np.random.randn(
            self.net.delta_manager.calculate_size())))
        out2 = self.net.forward_pass(self.X).as_array().copy()
        deltas2 = self.net.backward_pass(out2).as_array().copy()
        self.assertTrue(np.allclose(deltas1, deltas2))
