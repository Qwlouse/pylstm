#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


def approx_fprime(xk,f,epsilon,*args):
    f0 = f(*((xk,)+args))
    grad = np.zeros((len(xk),), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = epsilon
        grad[k] = (f(*((xk+ei,)+args)) - f0)/epsilon
        ei[k] = 0.0
    return grad


def check_gradient(net, X=None, T=None, n_timesteps=3, n_batches=5,
                   rnd=np.random.RandomState()):
    if X is None:
        X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    if T is None:
        T = rnd.randn(n_timesteps, n_batches, net.get_output_size())
        # normalize targets to sum to one
        T = T / T.sum(2).reshape(n_timesteps, n_batches, 1)

    weights = net.param_buffer.copy()

    ######### calculate gradient ##########
    net.forward_pass(X)
    net.backward_pass(T)
    grad_calc = net.calc_gradient().squeeze().copy()

    ######### estimate gradient ##########
    def f(W):
        net.param_buffer = W
        net.forward_pass(X)
        return net.calculate_error(T).copy()

    grad_approx = approx_fprime(weights.copy(), f, 1e-7)
    return np.sum((grad_approx - grad_calc) ** 2) / n_batches, grad_calc, grad_approx


def check_deltas(net, X=None, T=None, n_timesteps=3, n_batches=5,
                 rnd=np.random.RandomState()):
    if X is None:
        X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    if T is None:
        T = rnd.randn(n_timesteps, n_batches, net.get_output_size())
        # normalize targets to sum to one
        T = T / T.sum(2).reshape(n_timesteps, n_batches, 1)

    ######### calculate gradient ##########
    net.forward_pass(X)
    delta_calc = net.backward_pass(T).flatten()

    ######### estimate gradient ##########
    def f(X):
        net.forward_pass(X.reshape(n_timesteps, n_batches, -1))
        return net.calculate_error(T)

    delta_approx = approx_fprime(X.copy().flatten(), f, 1e-7)
    return np.sum((delta_approx - delta_calc) ** 2) / n_batches, delta_calc, delta_approx


def check_rpass(net, X=None, r=1e-7, n_timesteps=3, n_batches=5,
                rnd=np.random.RandomState()):
    if X is None:
        X = rnd.randn(n_timesteps, n_batches, net.get_input_size())
    nr_timesteps, nr_batches, _ = X.shape

    weights = net.param_buffer.copy()
    errs = np.zeros_like(weights)
    v = np.zeros_like(weights)
    for i in range(len(weights)):
        v[i] = 1.0
        net.param_buffer = weights
        out1 = net.forward_pass(X).copy()
        net.param_buffer = weights + r * v
        out2 = net.forward_pass(X)
        estimated = (out2 - out1) / r
        net.param_buffer = weights
        calculated = net.r_forward_pass(X, v)
        errs[i] = np.sum((estimated - calculated)**2)
        v[i] = 0.0

    return np.sum(errs**2), errs


def estimate_jacobian(net, X=None, nr_timesteps=4, nr_batches=5,
                      rnd=np.random.RandomState()):
    out_size = net.get_output_size()
    in_size = net.get_input_size()
    if X is None:
        X = rnd.randn(nr_timesteps, nr_batches, in_size)
    nr_timesteps, nr_batches, _ = X.shape

    out = net.forward_pass(X).copy()
    J = np.zeros((out_size * nr_timesteps * nr_batches, net.get_param_size()))

    for i in range(out_size * nr_timesteps * nr_batches):
        out_masked = np.zeros_like(out).flatten()
        out_masked[i] = 1.0
        out_masked = out_masked.reshape(*out.shape)

        deltas = net.pure_backpass(out_masked)
        J[i, :] = net.calc_gradient().flatten()
    return J


def check_gn_pass(net, X=None, v=None, r=1e-7, nr_timesteps=3, nr_batches=5,
                  rnd=np.random.RandomState()):
    out_size = net.get_output_size()
    in_size = net.get_input_size()
    if X is None:
        X = rnd.randn(nr_timesteps, nr_batches, in_size)
    nr_timesteps, nr_batches, _ = X.shape

    if v is None:
        v = rnd.randn(net.get_param_size())

    J = estimate_jacobian(net, X)
    G = J.T.dot(J)

    calc = net.hessian_pass(X, v).flatten()
    estimated = G.dot(v).flatten()
    return np.sum((calc - estimated) ** 2), calc, estimated


def construct_period_mask(periods):
    """
    Construct a mask for the recurrent matrix of an ArnnLayer, to ensure that
    connections only go to units of higher frequency, but not back.
    """
    unique_ps = sorted(set(periods))
    D = np.zeros((len(periods), len(periods)), dtype=np.float64)
    offset = 0
    for p in unique_ps:
        group_size = periods.count(p)
        D[offset:, offset:offset + group_size] = 1.0
        offset += group_size
    return D


def get_sequence_lengths(M):
    """
    Given a mask M it returns a list of the lengths of all sequences. Note: this
    assumes, that the mask has only values 0 and 1. It returns for each sequence
    the last index such that the mask is 1 there.
    :param M: mask of 0s and 1s with shape=(t, b, 1)
    :return: array of sequence lengths with shape=(b,)
    """
    return M.shape[0] - M[::-1, :, 0].argmax(axis=0)


################################################################################
# Best path decoding for monitoring Phoneme Errors

def ctc_best_path_decoding(Y):
    assert Y.shape[1] == 1
    Y_win = Y.argmax(2).reshape(Y.shape[0])
    t = []
    blank = True
    for y in Y_win:
        if blank is True and y != 0:
            t.append(y - 1)
            blank = False
        elif blank is False:
            if y == 0:
                blank = True
            elif y - 1 != t[-1]:
                t.append(y - 1)
    return t