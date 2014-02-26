#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import itertools
import unittest

import numpy as np
from pylstm import Gaussian, create_targets_object

from pylstm.structure import LstmLayer, Lstm97Layer, RnnLayer, MrnnLayer
from pylstm.structure import build_net, ForwardLayer, InputLayer, LWTALayer
from pylstm.utils import check_gradient, check_deltas, check_rpass
from pylstm.wrapper import Matrix


rnd = np.random.RandomState(213998106)


class NetworkTests(unittest.TestCase):
    def build_network(self, layer_type, activation_function, layers=1):
        prev_layer = InputLayer(self.input_size)
        for l in range(layers):
            prev_layer = prev_layer >> layer_type(self.output_size, act_func=activation_function)
        net = build_net(prev_layer)
        net.initialize(Gaussian(std=0.1))
        return net

    def build_lwta_network(self, input_size, activation_function, block_sizes=[1, 2, 4, 8]):
        prev_layer = InputLayer(input_size)
        for l in range(len(block_sizes)):
            prev_layer = prev_layer >> ForwardLayer(input_size, act_func=activation_function)
            prev_layer = prev_layer >> LWTALayer(block_size=block_sizes[l])
        net = build_net(prev_layer)
        net.initialize(Gaussian(std=0.1))
        return net

    def setUp(self):
        self.input_size = 2
        self.output_size = 3
        self.layer_types = [ForwardLayer, RnnLayer, MrnnLayer, LstmLayer, Lstm97Layer]
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
        targets = create_targets_object(np.zeros_like(out1))
        deltas1 = net.backward_pass(targets).copy()
        bwstate1 = net.get_bwd_state_for('LstmLayer')
        b1 = {}
        for h in bwstate1.keys():
            b1[h] = bwstate1[h].copy()

        net.bwd_state_manager.initialize_buffer(Matrix(rnd.randn(
            net.bwd_state_manager.calculate_size())))
        net.forward_pass(self.X).copy()
        deltas2 = net.backward_pass(targets).copy()
        bwstate2 = net.get_bwd_state_for('LstmLayer')
        b2 = {}
        for h in bwstate2.keys():
            b2[h] = bwstate2[h].copy()

        for b in b2:
            print(b)
            print(b1[b] - b2[b])

        self.assertTrue(np.allclose(deltas1, deltas2))

    def test_deltas_finite_differences(self):
        t = 7
        b = 5
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, grad_calc, grad_approx = check_deltas(net, n_batches=b,
                                                     n_timesteps=t, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                diff = (grad_approx - grad_calc).reshape(t, b, -1)
                for t in range(diff.shape[0]):
                    print("======== t=%d =========" % t)
                    print(diff[t])
            print("Checking Deltas of %s with %s = %0.4f" % (l(3), a, e))

        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_gradient_finite_differences(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, grad_calc, grad_approx = check_gradient(net, n_batches=5,
                                                       n_timesteps=7, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(grad_approx - grad_calc)
                diff = layer.create_param_view(b)
                for n, q in diff.items():
                    print("====== %s ======" % n)
                    print(q)

            print("Checking Gradient of %s with %s = %0.4f" % (l(3), a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_lwta_gradient_finite_differences(self):
        check_errors = []
        for a in self.activation_functions:
            net = self.build_lwta_network(8, a)
            e, grad_calc, grad_approx = check_gradient(net, n_batches=5,
                                                       n_timesteps=7, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(grad_approx - grad_calc)
                diff = layer.create_param_view(b)
                for n, q in diff.items():
                    print("====== %s ======" % n)
                    print(q)

            print("Checking Gradient of %s with LWTA = %0.4f" % (a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_rforward_finite_differences(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            e, allerrors = check_rpass(net, n_batches=5, n_timesteps=2, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(allerrors.copy())
                diff = layer.create_param_view(b)
                for n, q in diff.items():
                    print("====== %s ======" % n)
                    print(q)

            print("Checking RForward pass of %s with %s = %0.4g" % (l(3), a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_rforward_finite_differences_multilayer(self):
        check_errors = []
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a, layers=2)
            e, allerrors = check_rpass(net, n_batches=5, n_timesteps=7, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(allerrors.copy())
                diff = layer.create_param_view(b)
                for n, q in diff.items():
                    print("====== %s ======" % n)
                    print(q)

            print("Checking RForward pass of %s with %s = %0.4g" % (l(3), a, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))
