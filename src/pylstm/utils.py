#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from pylstm.layers import NpFwdLayer
from pylstm.netbuilder import NetworkBuilder
from scipy.optimize import approx_fprime
import numpy as np



def check_deltas(net, X = None):
    def f(x):
        out = net.forward_pass(x.reshape(1,1,-1)).as_array()
        return .5*np.sum(out**2)

    if X is None:
        X = np.random.randn(net.get_input_size())

    delta_approx = approx_fprime(X, f, 1e-7)
    out = net.forward_pass(X.reshape(1,1,-1)).as_array()
    delta_calc = net.backward_pass(out).as_array()
    return np.sum((delta_approx - delta_calc)**2), delta_calc, delta_approx



if __name__ == "__main__":
    netb  = NetworkBuilder()
    netb.input(5) >> NpFwdLayer(7) >> netb.output
    net = netb.build()
    weights = np.random.randn(net.get_param_size())
    # and set them as the parameter buffer
    net.set_param_buffer(weights)

    print(check_deltas(net))




