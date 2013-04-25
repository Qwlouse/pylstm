#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
#from scipy.optimize import fmin_ncg
from conjgrad import fmin_ncg
import sys
sys.path.append('.')
sys.path.append('..')

from pylstm.error_functions import MeanSquaredError
from pylstm import wrapper

rnd = np.random.RandomState()


def print_error_per_epoch(epoch, error):
    print("Epoch %d:\tError = %0.4f" % (epoch, error))


class SgdTrainer(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.param_buffer
        for epoch in range(1, epochs + 1):
            net.forward_pass(X)
            error = net.calculate_error(T)
            callback(epoch, error)
            net.backward_pass(T)
            grad = net.calc_gradient()
            grad *= - self.learning_rate
            weights += grad


class RPropTrainer(object):
    def __init__(self, learning_rate=0.1, error_fkt=MeanSquaredError):
        self.learning_rate = learning_rate
        self.error_fkt = error_fkt()
        self.initialized = False

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.param_buffer
        for epoch in range(1, epochs + 1):
            out = net.forward_pass(X)
            error = self.error_fkt.forward_pass(out, T) / X.shape[1]
            callback(epoch, error)
            deltas = self.error_fkt.backward_pass(out, T)
            net.backward_pass(deltas)
            grad = net.calc_gradient()

 
            #calculate grad sign
            grad_sign = (grad > 0.0)
 
            if not self.initialized:
                self.last_grad_sign = grad_sign
                self.stepsize = np.ones_like(grad_sign) * .00001
                self.initialized = True
                continue
            increase = (grad_sign == self.last_grad_sign)
            self.stepsize = (self.stepsize * (increase * 1.01 + (increase == False) * .99))

            grad[:] = self.stepsize * grad_sign + -self.stepsize * (grad_sign == False)
            #print("grad arr:", grad_arr)
            #print("grad:", grad)
            #print(((grad_sign==False)).flatten())
            print("weights before:", weights.flatten())
            weights += grad
            print("weights after:", weights.flatten())
            self.last_grad_sign = grad_sign.copy()
            

class CgTrainer(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.param_buffer.copy()
        lambda_ = .1
        rho = 1.0 / 30.0
        #run forward pass, output saved in out
        net.param_buffer = weights
        out = net.forward_pass(X)

        #calculate error
        error = net.calculate_error(T) / X.shape[1]
        callback(0, error)

        net.backward_pass(T)
        grad = 1*net.calc_gradient()


        for epoch in range(1, epochs + 1):
            
            #select an input batch, and target batch
            
            #run forward pass, output saved in out
            net.param_buffer = weights
            out = net.forward_pass(X)
            
            #calculate error
            error = net.calculate_error(T) / X.shape[1]
            callback(epoch, error)

            net.backward_pass(T)
            grad = net.calc_gradient()

            #initialize v, but maybe we should use the small random numbers like in old version
            v = np.zeros(net.get_param_size())

            #run cg
            def f(W):
                # net.param_buffer = W
                net.forward_pass(X)
                return net.calculate_error(T)

            def fprime(W):
                net.param_buffer = W
                net.forward_pass(X)
                net.backward_pass(T)
                return net.calc_gradient().copy().flatten()

            def fhess_p(v):
                #net.param_buffer = weights.as_array.copy()
                return net.hessian_pass(X, v, lambda_=.1, mu=1.0/30.0).copy().flatten() + lambda_ * v

            xopt, allvecs = fmin_ncg(f, v, grad, fhess_p=fhess_p, maxiter=150, retall=True, disp=True)


            ## backtrack #1
            prevError = error
            idx = len(allvecs)-1
            lowError = float('Inf')
            for testW in reversed(allvecs):
                net.param_buffer = weights.copy() + testW
                out = net.forward_pass(X)
                tmpError = net.calculate_error(T) / X.shape[1]
                if tmpError < lowError:
                    lowError = tmpError
                    lowIdx = idx
                idx -= 1

            bestDW = allvecs[lowIdx]

            ## backtrack #2
            finalDW = bestDW
            for j in range(1,10):
                net.param_buffer = weights.copy() + .9**j * bestDW
                out = net.forward_pass(X)
                tmpError = net.calculate_error(T) / X.shape[1]
                if tmpError < lowError:
                    finalDW = .9**j * bestDW
                    lowError = tmpError

            ## Levenberg-Marquardt heuristic
            drop = 2.0 / 3.0
            boost = 1/drop
            denom = 0.5*(np.dot(finalDW, fhess_p(finalDW)) + np.dot(np.squeeze(grad), finalDW))
            rho = (lowError - prevError)/denom
            if rho < 0.25:
                lambda_ = lambda_ * boost
            elif rho > 0.75:
                   lambda_ = lambda_ * drop


            weights = weights.copy() + finalDW

            #xopt, allvecs = fmin_ncg(f, np.zeros_like(weights), fprime, fhess_p=fhess_p, maxiter=50, retall=True, disp=True)


            # #dws = cg(v, grad, lambda, mu)
            #
            # #but can we do this backwards
            # for dwvec in dws:
            #     tmp_weights = weights.copy() + dwvec
            #     net.param_buffer = tmp_weight.copy()
            #     tmp_out = net.forward_pass(X)
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
            # net.param_buffer = tmp_weight.copy()
            # tmp_out = net.forward_pass(X)
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
            #grad *= - self.learning_rate
            #weights += grad


if __name__ == "__main__":
    from netbuilder import NetworkBuilder
    from layers import LstmLayer, RegularLayer

    numbatches = 1
    numtimesteps = 5
    numIn = 4
    numOut = 3

    netb = NetworkBuilder()
    netb.input(4) >> LstmLayer(3) >> netb.output
    net = netb.build()
    weight = rnd.randn(net.get_param_size())
    net.param_buffer = weight.copy()
    trainer = CgTrainer(learning_rate=0.01)
    X = rnd.randn(numtimesteps, numbatches,  numIn)
    T = rnd.randn(numtimesteps, numbatches, numOut)
    trainer.train(net, X, T, epochs=10)
