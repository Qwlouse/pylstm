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

def check_deltas(net, X = None):
    def f(x):
        net.clear_internal_state()
        out = net.forward_pass(x.reshape(1,1,-1)).as_array()

        return .5*np.sum(out**2)

    if X is None:
        X = rnd.randn(net.get_input_size())

    delta_approx = approx_fprime(X, f, 1e-1)
    out = net.forward_pass(X.reshape(1,1,-1)).as_array()
    delta_calc = net.backward_pass(out).as_array()
    return np.sum((delta_approx - delta_calc)**2), delta_calc, delta_approx



if __name__ == "__main__":
    netb  = NetworkBuilder()
    netb.input(5) >> LstmLayer(7) >> netb.output
    net = netb.build()

    weights = rnd.randn(net.get_param_size())
    # and set them as the parameter buffer
    net.set_param_buffer(weights)

    print(check_deltas(net))




