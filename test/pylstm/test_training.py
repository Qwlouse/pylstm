#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest

import numpy as np
from pylstm import Gaussian

from pylstm.structure import LstmLayer, Lstm97Layer, RnnLayer, MrnnLayer
from pylstm.structure import build_net, ForwardLayer, InputLayer
from pylstm.training import (ForwardStep, SgdStep, MomentumStep, NesterovStep,
                             RmsPropStep, RPropStep, Trainer, Minibatches,
                             MaxEpochsSeen)


rnd = np.random.RandomState(213998106)


class TrainingTests(unittest.TestCase):
    def build_network(self, layer_type, activation_function, layers=1):
        prev_layer = InputLayer(self.input_size)
        for l in range(layers):
            prev_layer = prev_layer >> layer_type(self.output_size, act_func=activation_function)
        net = build_net(prev_layer)
        net.initialize(Gaussian(std=0.1))
        return net

    def setUp(self):
        self.input_size = 2
        self.output_size = 3
        self.layer_types = [ForwardLayer, RnnLayer, MrnnLayer, LstmLayer, Lstm97Layer]
        self.activation_functions = ["linear", "tanh", "tanhx2", "sigmoid", "softmax"]
        n_timesteps = 5
        n_batches = 6
        self.X = rnd.randn(n_timesteps, n_batches, self.input_size)
        self.T = rnd.randn(n_timesteps, n_batches, self.output_size)
        self.T = self.T / self.T.sum(2).reshape(n_timesteps, n_batches, 1)

    def test_steps(self):
        """
        This test checks if one epoch of training works for a tanh LSTM network for all types of steps.
        """
        steps = (ForwardStep(), SgdStep(), MomentumStep(), NesterovStep(), RmsPropStep(), RPropStep())
        net = self.build_network(LstmLayer, "tanh")

        for step in steps:
            trainer = Trainer(net, step)
            trainer.stopping_criteria.append(MaxEpochsSeen(max_epochs=1))
            trainer.train(Minibatches(self.X, self.T, batch_size=4, verbose=False), verbose=False)

        # This test passes as long as training succeeds
