#!/usr/bin/env python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest

import numpy as np
from pylstm import Gaussian, create_targets_object

from pylstm.structure import Lstm97Layer
from pylstm.structure import build_net, InputLayer
from pylstm.utils import check_gradient, check_deltas, check_rpass
from pylstm.wrapper import Matrix


rnd = np.random.RandomState(213998106)


class NetworkTests(unittest.TestCase):
    def build_network(self, lstm_configuration):
        lstm_config = {
            'act_func': 'linear',
            'input_gate': True,
            'output_gate': True,
            'forget_gate': True,
            'peephole_connections': True,
            'gate_recurrence': False,
            'use_bias': True,
            'full_gradient': True
        }
        lstm_config.update(lstm_configuration)
        net = build_net(InputLayer(self.input_size) >> Lstm97Layer(self.output_size, **lstm_config))
        net.initialize(Gaussian(std=0.1))
        #net.initialize(1)
        return net

    def setUp(self):
        self.input_size = 3
        self.output_size = 2
        self.lstm_configs = [
            {'input_gate': False},
            {'output_gate': False},
            {'forget_gate': False},
            {'peephole_connections': False},
            {'use_bias': False},
            {'gate_recurrence': True},
        ]
        self.X = rnd.randn(10, 7, self.input_size)

    def test_lstm_forward_pass_insensitive_to_fwd_state(self):
        net = self.build_network({})
        out1 = net.forward_pass(self.X).copy()
        net.fwd_state_manager.initialize_buffer(Matrix(rnd.randn(
            net.fwd_state_manager.calculate_size())))
        out2 = net.forward_pass(self.X).copy()
        self.assertTrue(np.allclose(out1, out2))

    def test_lstm_backward_pass_insensitive_to_bwd_state(self):
        net = self.build_network({})
        net.clear_internal_state()
        out1 = net.forward_pass(self.X).copy()
        targets = create_targets_object(np.zeros_like(out1))
        deltas1 = net.backward_pass(targets).copy()
        bwstate1 = net.get_bwd_state_for('Lstm97Layer')
        b1 = {}
        for h in bwstate1.keys():
            b1[h] = bwstate1[h].copy()

        net.bwd_state_manager.initialize_buffer(Matrix(rnd.randn(
            net.bwd_state_manager.calculate_size())))
        net.forward_pass(self.X).copy()
        deltas2 = net.backward_pass(targets).copy()
        bwstate2 = net.get_bwd_state_for('Lstm97Layer')
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
        for cfg in self.lstm_configs:
            net = self.build_network(cfg)
            e, grad_calc, grad_approx = check_deltas(net, n_batches=b,
                                                     n_timesteps=t, rnd=rnd)
            check_errors.append(e)
            if e > 1e-6:
                diff = (grad_approx - grad_calc).reshape(t, b, -1)
                for t in range(diff.shape[0]):
                    print("======== t=%d =========" % t)
                    print(diff[t])
            print("Checking Deltas of Lstm97 with %s = %0.6f" % (cfg, e))

        self.assertTrue(np.all(np.array(check_errors) < 1e-6))

    def test_gradient_finite_differences(self):
        check_errors = []
        for cfg in self.lstm_configs:
            net = self.build_network(cfg)
            e, grad_calc, grad_approx = check_gradient(net, n_batches=10,
                                                       n_timesteps=10, rnd=rnd)
            check_errors.append(e)
            if e > 1e-4:
                # construct a weight view and break down the differences
                layer = net.layers.values()[1]  # the only layer
                b = Matrix(grad_approx - grad_calc)
                a = Matrix(grad_approx)
                c = Matrix(grad_calc)
                # appr = layer.create_param_view(a)
                # calc = layer.create_param_view(c)
                diff = layer.create_param_view(b)
                for n, q in diff.items():
                    print("====== %s ======" % n)
                    # print(appr[n])
                    # print(calc[n])
                    print(q)

            print("Checking Gradient of Lstm97 with %s = %0.4f" % (cfg, e))
        self.assertTrue(np.all(np.array(check_errors) < 1e-4))

    # def test_rforward_finite_differences(self):
    #     check_errors = []
    #     for cfg in self.lstm_configs:
    #         net = self.build_network(cfg)
    #         e, allerrors = check_rpass(net, n_batches=5, n_timesteps=2, rnd=rnd)
    #         check_errors.append(e)
    #         if e > 1e-4:
    #             # construct a weight view and break down the differences
    #             layer = net.layers.values()[1]  # the only layer
    #             b = Matrix(allerrors.copy())
    #             diff = layer.create_param_view(b)
    #             for n, q in diff.items():
    #                 print("====== %s ======" % n)
    #                 print(q)
    #
    #         print("Checking RForward pass of Lstm97 with %s = %0.4g" % (cfg, e))
    #     self.assertTrue(np.all(np.array(check_errors) < 1e-4))
