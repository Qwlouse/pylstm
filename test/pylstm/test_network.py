#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
import numpy as np
import itertools
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import LstmLayer, RnnLayer, RegularLayer
from pylstm.utils import check_gradient, check_deltas
from pylstm.wrapper import Matrix

rnd = np.random.RandomState(2634587)


def check_rpass(net, weights, v, r=1e-7):
    n_timesteps = 2
    n_batches = 1
    X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    T = np.zeros((n_timesteps, n_batches, net.get_output_size()))
    T[:, :, 0] = 1.0  # so the outputs sum to one
    net.param_buffer = weights
    out1 = net.forward_pass(X).copy()
    net.param_buffer = weights + r * v
    out2 = net.forward_pass(X)
    estimated = (out2 - out1) / r
    net.param_buffer = weights
    calculated = net.r_forward_pass(X, v)
    return np.sum((estimated - calculated)**2), calculated, estimated


def check_rpass_full(net):
    weights = rnd.randn(net.get_param_size())
    errs = np.zeros_like(weights)
    for i in range(len(weights)):
        v = np.zeros_like(weights)
        v[i] = 1.0
        errs[i], calc, est = check_rpass(net, weights, v)
    return np.sum(errs**2), errs


class NetworkTests(unittest.TestCase):
    def build_network(self, layer_type, activation_function, layers=1):
        netb = NetworkBuilder()

        prev_layer = netb.input(self.input_size)
        for l in range(layers):
            prev_layer = prev_layer >> layer_type(self.output_size, act_func=activation_function)
        prev_layer >> netb.output

        net = netb.build()
        net.param_buffer = rnd.randn(net.get_param_size())
        return net

    def setUp(self):
        self.input_size = 2
        self.output_size = 3
        self.layer_types = [RegularLayer, RnnLayer, LstmLayer]
        self.activation_functions = ["linear", "tanh", "tanhx2", "sigmoid", "softmax"]
        self.X = rnd.randn(2, 7, self.input_size)

    def test_lstm_forward_pass_insensitive_to_fwd_state(self):
        net = self.build_network(LstmLayer, "tanh")
        out1 = net.forward_pass(self.X).copy()
        net.fwd_state_manager.initialize_buffer(Matrix(rnd.randn(
            net.fwd_state_manager.calculate_size())))
        out2 = net.forward_pass(self.X).copy()
        self.assertTrue(np.allclose(out1, out2))

    def test_lstm_backward_pass_insensitive_to_bwd_state(self):
        net = self.build_network(LstmLayer, "tanh")
        net.clear_internal_state()
        out1 = net.forward_pass(self.X).copy()
        deltas1 = net.backward_pass(np.zeros_like(out1)).copy()
        bwstate1 = net.get_bwd_state_for('LstmLayer')
        b1 = {}
        for h in bwstate1.keys():
            b1[h] = bwstate1[h].copy()

        net.bwd_state_manager.initialize_buffer(Matrix(rnd.randn(
            net.bwd_state_manager.calculate_size())))
        out2 = net.forward_pass(self.X).copy()
        deltas2 = net.backward_pass(np.zeros_like(out2)).copy()
        bwstate2 = net.get_bwd_state_for('LstmLayer')
        b2 = {}
        for h in bwstate2.keys():
            b2[h] = bwstate2[h].copy()

        for b in b2:
            print(b)
            print(b1[b] - b2[b])

        self.assertTrue(np.allclose(deltas1, deltas2))

    def test_deltas_finite_differences(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, grad_calc, grad_approx = check_deltas(net, n_batches=5, n_timesteps=7)
            check_errors.append(e)
            if e > 1e-4:
                diff = (grad_approx - grad_calc).reshape(3, 3, -1)
                for t in range(diff.shape[0]):
                    print("======== t=%d =========" % t)
                    print(diff[t])
            print("Checking Deltas of %s with %s = %0.4f" % (l(3), a, e))

        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_gradient_finite_differences(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, grad_calc, grad_approx = check_gradient(net, n_batches=5, n_timesteps=7)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(grad_approx - grad_calc)
                diff = layer.create_param_view(b)
                for n, b in diff.items():
                    print("====== %s ======" % n)
                    print(b)

            print("Checking Gradient of %s with %s = %0.4f" % (l(3), a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_rforward_finite_differences(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, allerrors = check_rpass_full(net)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(allerrors.copy())
                diff = layer.create_param_view(b)
                for n, b in diff.items():
                    print("====== %s ======" % n)
                    print(b)

            print("Checking RForward pass of %s with %s = %0.4g" % (l(3), a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_rforward_finite_differences_multilayer(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a, layers=2)
            e, allerrors = check_rpass_full(net)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(allerrors.copy())
                diff = layer.create_param_view(b)
                for n, b in diff.items():
                    print("====== %s ======" % n)
                    print(b)

            print("Checking RForward pass of %s with %s = %0.4g" % (l(3), a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))