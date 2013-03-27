#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.optimize import fmin_ncg
import sys
sys.path.append('.')
sys.path.append('..')

from pylstm.error_functions import MeanSquaredError
from pylstm import wrapper

rnd = np.random.RandomState(12345)


def print_error_per_epoch(epoch, error):
    print("Epoch %d:\tError = %0.4f" % (epoch, error))


class SgdTrainer(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.get_param_buffer()
        for epoch in range(1, epochs + 1):
            net.forward_pass(X)
            error = net.calculate_error(T)
            callback(epoch, error)
            net.backward_pass(T)
            grad = net.calc_gradient()
            grad_arr = grad.as_array()
            grad_arr *= - self.learning_rate
            wrapper.add_into_b(grad, weights)


class RPropTrainer(object):
    def __init__(self, learning_rate=0.1, error_fkt=MeanSquaredError):
        self.learning_rate = learning_rate
        self.error_fkt = error_fkt()
        self.initialized = False

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.get_param_buffer()
        for epoch in range(1, epochs + 1):
            out = net.forward_pass(X).as_array()
            error = self.error_fkt.forward_pass(out, T) / X.shape[1]
            callback(epoch, error)
            deltas = self.error_fkt.backward_pass(out, T)
            net.backward_pass(deltas)
            grad = net.calc_gradient()
            grad_arr = grad.as_array()
 
            #calculate grad sign
            grad_sign = (grad_arr > 0.0)
 
            if not self.initialized:
                self.last_grad_sign = grad_sign
                self.stepsize = np.ones_like(grad_sign) * .00001
                self.initialized = True
                continue
            increase = (grad_sign == self.last_grad_sign)
            self.stepsize = (self.stepsize * (increase * 1.01 + (increase == False) * .99))

            grad_arr[:] = self.stepsize * grad_sign + -self.stepsize * (grad_sign == False)
            #print("grad arr:", grad_arr)
            #print("grad:", grad.as_array())
            #print(((grad_sign==False)).flatten())
            print("weights before:", weights.as_array().flatten())
            wrapper.add_into_b(grad, weights)
            print("weights after:", weights.as_array().flatten())
            self.last_grad_sign = grad_sign.copy()
            

class CgTrainer(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.get_param_buffer()
        for epoch in range(1, epochs + 1):
            
            #select an input batch, and target batch
            
            #run forward pass, output saved in out
            out = net.forward_pass(X).as_array()
            
            #calculate error
            error = net.calculate_error(T) / X.shape[1]
            callback(epoch, error)

            net.backward_pass(T)
            grad = net.calc_gradient()

            #initialize v, but maybe we should use the small random numbers like in old version
            v = np.zeros(net.get_param_size())

            #run cg
            def f(W):
                net.set_param_buffer(W)
                net.forward_pass(X)
                return net.calculate_error(T)

            def fprime(W):
                net.set_param_buffer(W)
                net.forward_pass(X)
                net.backward_pass(T)
                return net.calc_gradient().as_array()

            def fhess_p(W, v):
                net.set_param_buffer(W)
                return net.hessian_pass(X, v, lambda_=0., mu=0.).as_array()

            xopt, allvecs = fmin_ncg(f, weights.as_array().copy(), fprime, fhess_p=fhess_p, maxiter=50, retall=True, disp=True)
            # #dws = cg(v, grad, lambda, mu)
            #
            # #but can we do this backwards
            # for dwvec in dws:
            #     tmp_weights = weights.copy() + dwvec
            #     net.set_param_buffer(tmp_weight.copy())
            #     tmp_out = net.forward_pass(X).as_array()
            #     tmp_error = self.error_fkt.forward_pass(out,T) / X.shape[1]
            #
            #     if last_error > tmp_error:
            #         if n < dws.length() - 1:
            #             dw = dws[n + 1]
            #             break;
            #
            #     track_last_error = track_new_error;
            #
            # #Calculate rho based on dw
            # tmp_weights = weights.copy() + dw
            # net.set_param_buffer(tmp_weight.copy())
            # tmp_out = net.forward_pass(X).as_array()
            # new_error = self.error_fkt.forward_pass(out,T) / X.shape[1]
            #
            # #f_val = cg.f_val?!
            #
            # rho = (new_error - last_error)/f_val
            # if rho > .75:
            #     lambda_ *= 2/3
            # elif rho < .25:
            #     lambda_ *= 3/2
            
            
            #run backtrack 2 on dw

            #grad_arr = grad.as_array()
            #grad_arr *= - self.learning_rate
            #wrapper.add_into_b(grad, weights)


if __name__ == "__main__":
    from netbuilder import NetworkBuilder
    from layers import LstmLayer, RegularLayer
    netb = NetworkBuilder()
    netb.input(4) >> LstmLayer(3) >> netb.output
    net = netb.build()
    weight = rnd.randn(net.get_param_size())
    net.set_param_buffer(weight.copy())
    trainer = CgTrainer(learning_rate=0.01)
    X = rnd.randn(2, 5, 4)
    T = rnd.randn(2, 5, 3)
    trainer.train(net, X, T, epochs=10)
