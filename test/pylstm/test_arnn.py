#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import unittest
import numpy as np
import itertools
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import ARNN
from pylstm.utils import check_gradient, check_deltas, check_rpass
from pylstm.wrapper import Matrix

rnd = np.random.RandomState(213998106)


class ArnnTests(unittest.TestCase):
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
        self.timesteps = 10

        self.batch_size = 4
        netb = NetworkBuilder()
        netb.input(self.input_size) >> ARNN(self.output_size) >> netb.output
        self.net = netb.build()
        self.net.param_buffer = np.ones(self.net.get_param_size())*2 #rnd.randn(self.net.get_param_size()) * 0.1
        self.net.get_param_view_for('ARNN')['Timing'][:] = [1, 2, 3]
        self.X = rnd.randn(self.timesteps, self.batch_size, self.input_size)


    def test_deltas_finite_differences(self):

        e, grad_calc, grad_approx = check_deltas(self.net, n_batches=self.batch_size,
                                                 n_timesteps=self.timesteps, rnd=rnd)

        grad_approx = grad_approx.reshape(self.timesteps, self.batch_size, -1)
        grad_calc = grad_calc.reshape(self.timesteps, self.batch_size, -1)
        diff = (grad_approx - grad_calc)
        for t in range(diff.shape[0]):
            print("======== t=%d =========" % t)
            print("calc")
            print(grad_calc[t])
            print("approx")
            print(grad_approx[t])
            print("diff")
            print(diff[t])
            print(np.sum(diff[t]**2))
        print("Checking Deltas of ARNN with sigmoid = %0.4f" % e)

        self.assertTrue(e < 1e-6)

    def test_gradient_finite_differences(self):
        e, grad_calc, grad_approx = check_gradient(self.net, n_batches=self.batch_size,
                                                 n_timesteps=self.timesteps, rnd=rnd)
        # construct a weight view and break down the differences
        layer = self.net.layers.values()[1]  # the only layer
        a = Matrix(grad_approx)
        b = Matrix(grad_approx - grad_calc)
        c = Matrix(grad_calc)

        diff = layer.create_param_view(b)
        approx = layer.create_param_view(a)
        calc = layer.create_param_view(c)
        E = 0.0

        for n, q in diff.items():
            if n == 'Timing':
                continue
            print("====== %s ======" % n)
            print("Calculated:")
            print(calc[n])
            print("Approx:")
            print(approx[n])
            print("Difference:")
            print(q)

            err = np.sum(q ** 2) / self.batch_size
            print(err)
            E += err

        print("Checking Gradient of ARNN with sigmoid = %0.4f" % E)
        self.assertTrue(E < 1e-6)


"""
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
"""