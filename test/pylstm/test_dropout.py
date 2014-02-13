#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import itertools
import unittest

import numpy as np
from pylstm import Gaussian

from pylstm.structure import LstmLayer, Lstm97Layer, RnnLayer, MrnnLayer, DropoutLayer
from pylstm.structure import build_net, ForwardLayer, InputLayer
from pylstm.utils import check_gradient, check_deltas, check_rpass
from pylstm.wrapper import Matrix


rnd = np.random.RandomState(213998106)


class NetworkTests(unittest.TestCase):
    def build_network(self, layer_type, activation_function, layers=1):
        prev_layer = InputLayer(self.input_size)
        prev_layer = prev_layer >> DropoutLayer(dropout_prob=0.2)
        for l in range(layers):
            prev_layer = prev_layer >> layer_type(self.output_size, act_func=activation_function)
            prev_layer = prev_layer >> DropoutLayer(dropout_prob=0.5)
        prev_layer = prev_layer >> ForwardLayer(self.output_size, act_func="softmax")
        net = build_net(prev_layer)
        net.initialize(Gaussian(std=0.1))
        return net

    def setUp(self):
        self.input_size = 10
        self.output_size = 3
        self.layer_types = [ForwardLayer, RnnLayer, MrnnLayer, LstmLayer, Lstm97Layer]
        self.activation_functions = ["linear", "tanh", "tanhx2", "sigmoid"]
        self.X = rnd.randn(2, 7, self.input_size)

    def test_dropout_mask_application(self):
        net = build_net(InputLayer(self.input_size) >> DropoutLayer())
        output = net.forward_pass(self.X)
        mask = net.get_fwd_state_for('DropoutLayer')['Mask']
        self.assertTrue(np.all(output[mask == 0] == 0))

    def test_dropout_layer_fwd_pass(self):
        for l, a in itertools.product(self.layer_types, self.activation_functions):
            net = self.build_network(l, a)
            output = net.forward_pass(self.X)

        # This test passes if forward pass succeeds
