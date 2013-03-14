#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
import numpy as np
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import LstmLayer
from pylstm.trainer import MeanSquaredError
from pylstm.wrapper import Buffer
from scipy.optimize import approx_fprime
rnd = np.random.RandomState(43210)


def check_gradient(net):
    n_timesteps = 15
    n_batches = 10
    X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    #X = np.ones((n_timesteps, n_batches, net.get_input_size()))
    T = np.zeros((n_timesteps, n_batches, net.get_output_size()))
    error_fkt = MeanSquaredError()
    weights = rnd.randn(net.get_param_size())
    #weights = np.ones((net.get_param_size()))
    net.set_param_buffer(weights.copy())

    ######### calculate gradient ##########
    net.forward_pass(X).as_array()
    net.backward_pass(T)
    grad_calc = net.calc_gradient().as_array().squeeze()

    ######### estimate gradient ##########
    def f(W):
        net.set_param_buffer(W)
        net.forward_pass(X)
        return net.calculate_error(T)

    grad_approx = approx_fprime(weights.copy(), f, 1e-7)
    return np.sum((grad_approx - grad_calc) ** 2) / n_batches, grad_calc, grad_approx


class NetworkTests(unittest.TestCase):
    def setUp(self):
        netb = NetworkBuilder()
        netb.input(5) >> LstmLayer(3) >> netb.output
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

    def test_gradient_finite_differences(self):
        e, grad_calc, grad_approx = check_gradient(self.net)
        self.assertLess(e, 1e-4)
