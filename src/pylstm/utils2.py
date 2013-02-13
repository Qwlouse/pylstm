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


def check_gradient(net):
    timesteps = 5
    n_batches = 5

    X = np.ones((timesteps, n_batches, net.get_input_size()))
    #X = rnd.randn(timesteps, n_batches, net.get_input_size())
    weights = np.ones(net.get_param_size())
    #weights = rnd.randn(net.get_param_size())
    net.set_param_buffer(weights)

    out = net.forward_pass(X).as_array()
    
    delta_calc = net.backward_pass(out).as_array()
    grad_calc = net.calc_gradient().as_array()

    def f(w):
        net.set_param_buffer(w)
        out = net.forward_pass(X).as_array()
        return .5 * np.sum(out ** 2)#/n_batches

    grad_approx = approx_fprime(weights, f, 1e-7)
    return np.sum((grad_approx - grad_calc.squeeze()) ** 2), grad_calc, grad_approx



if __name__ == "__main__":
    netb  = NetworkBuilder()
    netb.input(2) >> LstmLayer(3) >> netb.output
    net = netb.build()

    err, grad_calc, grad_approx = check_gradient(net)

    print(grad_approx)
    print(grad_calc.squeeze())
    print(err)




