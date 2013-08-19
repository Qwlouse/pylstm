#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from .data_iterators import Minibatches


def conjgrad(gradient, v, f_hessp, maxiter=20):
    r = gradient - f_hessp(v)  # residual
    p = r  # current step
    rsold = r.T.dot(r)
    allvecs = [v.copy()]
    for i in range(maxiter):
        Ap = f_hessp(p)
        curv = (p.T.dot(Ap))
        if curv < 3 * np.finfo(np.float64).eps:
            break  # curvature is negative or zero
        alpha = rsold / curv
        v += alpha * p
        allvecs.append(v.copy())
        r = r - alpha * Ap   # updated residual
        rsnew = r.T.dot(r)
        if np.sqrt(rsnew) < 1e-10:
            break  # found solution
        p = r + rsnew / rsold * p

        rsold = rsnew

    return allvecs


class CgLiteTrainer(object):
    def __init__(self):
        pass

    def train(self, net, X, T, M=None, epochs=10, minibatch_size=32,
              mu=1. / 30, maxiter=20, success=lambda x: False):
        # TODO remove Loop=True
        mb = Minibatches(X, T, M, 1, loop=True)()
        lambda_ = .1

        for i in range(epochs):
            print("======= Epoch %d =====" % i)
            ## calculate the gradient
            grad = np.zeros_like(net.param_buffer.flatten())
            error = []
            for x, t, m in Minibatches(minibatch_size)(X, T, M):
                net.forward_pass(x)
                net.backward_pass(t, m)
                error.append(net.calculate_error(t, m))
                grad += net.calc_gradient().flatten()
            error = np.mean(error)
            print("Error:", error)

            ## initialize v
            v = np.zeros(net.get_param_size())

            ## get next minibatch to work with
            x, t, m = mb.next()

            ## define hessian pass
            def fhess_p(v):
                return net.hessian_pass(x, v, mu, lambda_).copy().flatten()

            ## run CG
            all_v = conjgrad(grad, v.copy(), fhess_p, maxiter=maxiter)

            ## backtrack #1
            lowError = float('Inf')
            lowIdx = 0
            weights = net.param_buffer.copy()
            for i, testW in reversed(list(enumerate(all_v))):
                net.param_buffer = weights - testW
                net.forward_pass(x)
                tmpError = net.calculate_error(t, m)
                if tmpError < lowError:
                    lowError = tmpError
                    lowIdx = i
            bestDW = all_v[lowIdx]

            ## backtrack #2
            finalDW = bestDW
            for j in np.arange(0, 1.0, 0.1):
                tmpDW = j * bestDW
                net.param_buffer = weights - tmpDW
                net.forward_pass(X)
                tmpError = net.calculate_error(T)
                if tmpError < lowError:
                    finalDW = tmpDW
                    lowError = tmpError

            ## Levenberg-Marquardt heuristic
            boost = 3.0 / 2.0
            net.param_buffer = weights
            denom = 0.5 * (np.dot(finalDW, fhess_p(finalDW))) + np.dot(
                np.squeeze(grad), finalDW) + error
            rho = (lowError - error) / denom
            if rho < 0.25:
                lambda_ *= boost
            elif rho > 0.75:
                lambda_ /= boost

            ## update weights
            net.param_buffer = weights - finalDW

            if success(net):
                print('Success!!!! after %d Epochs' % i)
                return i