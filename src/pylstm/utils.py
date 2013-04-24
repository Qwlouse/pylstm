#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import sys

sys.path.append('.')
sys.path.append('..')
from pylstm.layers import NpFwdLayer, LstmLayer
from pylstm.netbuilder import NetworkBuilder
from scipy.optimize import approx_fprime
import numpy as np

rnd = np.random.RandomState(12345)

def check_gradient(net, X=None):
    X = rnd.randn(net.get_input_size() * 2)

    def f(w):
        net.clear_internal_state()
        net.set_param_buffer(w)
        out = net.forward_pass(X.reshape(2, 1, -1))

        return .5 * np.sum(out ** 2)

    weights = rnd.randn(net.get_param_size())

    grad_approx = approx_fprime(weights, f, 1e-7)

    net.clear_internal_state()
    net.set_param_buffer(weights)
    out = net.forward_pass(X.reshape(2, 1, -1))
    delta_calc = net.backward_pass(out)
    grad_calc = net.calc_gradient()

    return np.sum((grad_approx - grad_calc) ** 2), grad_calc, grad_approx


if __name__ == "__main__":
    netb = NetworkBuilder()
    netb.input(5) >> LstmLayer(7) >> netb.output
    net = netb.build()

    #weights = rnd.randn(net.get_param_size())
    # and set them as the parameter buffer
    #net.set_param_buffer(weights)

    print(check_gradient(net))