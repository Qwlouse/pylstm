#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import itertools
import unittest

import numpy as np
from pylstm import Gaussian, create_targets_object

from pylstm.structure import LstmLayer, Lstm97Layer, RnnLayer, MrnnLayer, StaticLstmLayer, GatedLayer
from pylstm.structure import build_net, ForwardLayer, InputLayer, LWTALayer, ZeroLayer, NoOpLayer
from pylstm.utils import check_gradient, check_deltas, check_rpass
from pylstm.wrapper import Matrix


rnd = np.random.RandomState(213998106)


class GatedLayerTests(unittest.TestCase):

    def build_gatedlayer_network(self, input_size, activation_function):
        input_layer = InputLayer(input_size)
        hidden_layer = GatedLayer(input_size, act_func=activation_function)
        out_layer = GatedLayer(input_size, act_func=activation_function)

        input_layer >> hidden_layer >> out_layer
        net = build_net(input_layer)
        net.initialize(Gaussian(std=0.1))
        return net

    def setUp(self):
        self.input_size = 2
        self.output_size = 4
        self.layer_types = [ForwardLayer, RnnLayer, MrnnLayer, LstmLayer, Lstm97Layer]
        self.activation_functions = ["linear", "relu", "lwta", "tanh", "tanhx2", "sigmoid", "softmax"]
        self.X = rnd.randn(2, 7, self.input_size)

    def test_gatedlayer_deltas_finite_differences(self):
        t = 7
        b = 5
        check_errors = []
        for act_func in ['sigmoid', 'relu']:
            net = self.build_gatedlayer_network(3, act_func)
            e, grad_calc, grad_approx = check_deltas(net, n_batches=b,
                                                     n_timesteps=t, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                diff = (grad_approx - grad_calc).reshape(t, b, -1)
                for t in range(diff.shape[0]):
                    print("======== t=%d =========" % t)
                    print(diff[t])
            #print("Checking Deltas of %s with %s = %0.4f" % (l(3), a, e))

            self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    def test_gatedlayer_gradient_finite_differences(self):
        t = 7
        b = 5
        check_errors = []
        for act_func in ['sigmoid', 'relu']:
            net = self.build_gatedlayer_network(3, act_func)
            e, grad_calc, grad_approx = check_gradient(net, n_batches=b, n_timesteps=t, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(grad_approx - grad_calc)
                diff = layer.create_param_view(b)
                for n, q in diff.items():
                    print("====== %s ======" % n)
                    print(q)

            # print("Checking Gradient of %s with %s = %0.4f" % (l(3), a, e))
            self.assertTrue(np.all(np.array(check_errors) < 1e-4))
