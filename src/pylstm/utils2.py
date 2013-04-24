#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import sys
sys.path.append('.')
sys.path.append('..')
from pylstm.layers import  LstmLayer
from pylstm.netbuilder import NetworkBuilder
from scipy.optimize import approx_fprime
import numpy as np

rnd = np.random.RandomState(12345)


def check_gradient(net):
    timesteps = 15
    n_batches = 1

    #X = np.ones((timesteps, n_batches, net.get_input_size()))
    X = rnd.randn(timesteps, n_batches, net.get_input_size())
    weights = np.ones(net.get_param_size())
    #weights = rnd.randn(net.get_param_size())
    net.set_param_buffer(weights)
    IX = net.get_param_view_for('LstmLayer').IX.flatten()
    IX[:] = np.random.randn(len(IX))


    out = net.forward_pass(X)
    
    delta_calc = net.backward_pass(out)
    grad_calc = net.calc_gradient()

    def f(w):
        net.set_param_buffer(w)
        out = net.forward_pass(X)
        return .5 * np.sum(out ** 2)#/n_batches

    grad_approx = approx_fprime(weights, f, 1e-7)

    approx_error = np.sum((grad_approx - grad_calc.squeeze()) ** 2) / n_batches
    return approx_error, grad_calc, grad_approx


if __name__ == "__main__":
    netb  = NetworkBuilder()
    netb.input(2) >> LstmLayer(3) >> netb.output
    net = netb.build()

    err, grad_calc, grad_approx = check_gradient(net)

    print(grad_approx)
    print(grad_calc.squeeze())
    print(err)




