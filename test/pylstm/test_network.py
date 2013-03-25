#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
import numpy as np
import itertools
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import LstmLayer, RnnLayer, RegularLayer
from pylstm.wrapper import Buffer
from scipy.optimize import approx_fprime
rnd = np.random.RandomState(43210)


def check_gradient(net):
    n_timesteps = 3
    n_batches = 3
    X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    T = np.zeros((n_timesteps, n_batches, net.get_output_size()))
    T[:, :, 0] = 1.0  # so the outputs sum to one
    weights = rnd.randn(net.get_param_size())
    net.set_param_buffer(weights.copy())

    ######### calculate gradient ##########
    net.forward_pass(X)
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
    def build_network(self, layer_type, activation_function):
        netb = NetworkBuilder()
        netb.input(self.input_size) >> layer_type(self.output_size, act_func=activation_function) >> netb.output
        net = netb.build()
        net.set_param_buffer(np.random.randn(net.get_param_size()))
        return net

    def setUp(self):
        self.input_size = 5
        self.output_size = 3
        self.layer_types = [RegularLayer, RnnLayer, LstmLayer]
        self.activation_functions = ["linear", "tanh", "tanhx2", "sigmoid", "softmax"]
        self.X = np.random.randn(2, 7, self.input_size)

    def test_lstm_forward_pass_insensitive_to_internal_state(self):
        net = self.build_network(LstmLayer, "tanh")
        out1 = net.forward_pass(self.X).as_array().copy()
        net.intern_manager.initialize_buffer(Buffer(np.random.randn(
            net.intern_manager.calculate_size())))
        out2 = net.forward_pass(self.X).as_array().copy()
        self.assertTrue(np.allclose(out1, out2))

    def test_lstm_backward_pass_insensitive_to_internal_deltas(self):
        net = self.build_network(LstmLayer, "tanh")
        net.clear_internal_state()
        out1 = net.forward_pass(self.X).as_array().copy()
        deltas1 = net.backward_pass(out1).as_array().copy()
        net.intern_manager.initialize_buffer(Buffer(np.random.randn(
            net.intern_manager.calculate_size())))
        net.delta_manager.initialize_buffer(Buffer(np.random.randn(
            net.delta_manager.calculate_size())))
        out2 = net.forward_pass(self.X).as_array().copy()
        deltas2 = net.backward_pass(out2).as_array().copy()
        self.assertTrue(np.allclose(deltas1, deltas2))

    def test_gradient_finite_differences(self):
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, grad_calc, grad_approx = check_gradient(net)
            print("%s with %s" % (l(3), a))
            self.assertLess(e, 1e-4)
