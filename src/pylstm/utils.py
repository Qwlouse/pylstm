#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from scipy.optimize import approx_fprime
import numpy as np


def check_gradient(net, X=None, T=None, n_timesteps=3, n_batches=5, rnd=np.random.RandomState()):
    if X is None:
        X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    if T is None:
        T = rnd.randn(n_timesteps, n_batches, net.get_output_size())
        # normalize targets to sum to one
        T = T / T.sum(2).reshape(n_timesteps, n_batches, 1)

    weights = rnd.randn(net.get_param_size())
    net.param_buffer = weights.copy()

    ######### calculate gradient ##########
    net.forward_pass(X)
    net.backward_pass(T)
    grad_calc = net.calc_gradient().squeeze()

    ######### estimate gradient ##########
    def f(W):
        net.param_buffer = W
        net.forward_pass(X)
        return net.calculate_error(T)

    grad_approx = approx_fprime(weights.copy(), f, 1e-7)
    return np.sum((grad_approx - grad_calc) ** 2) / n_batches, grad_calc, grad_approx


def check_deltas(net, X=None, T=None, n_timesteps=3, n_batches=5, rnd=np.random.RandomState()):
    if X is None:
        X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    if T is None:
        T = rnd.randn(n_timesteps, n_batches, net.get_output_size())
        # normalize targets to sum to one
        T = T / T.sum(2).reshape(n_timesteps, n_batches, 1)

    weights = rnd.randn(net.get_param_size())
    net.param_buffer = weights.copy()

    ######### calculate gradient ##########
    net.forward_pass(X)
    delta_calc = net.backward_pass(T).flatten()

    ######### estimate gradient ##########
    def f(X):
        net.forward_pass(X.reshape(n_timesteps, n_batches, -1))
        return net.calculate_error(T)

    delta_approx = approx_fprime(X.copy().flatten(), f, 1e-7)
    return np.sum((delta_approx - delta_calc) ** 2) / n_batches, delta_calc, delta_approx
