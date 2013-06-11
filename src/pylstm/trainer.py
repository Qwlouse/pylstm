#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
#from scipy.optimize import fmin_ncg
from conjgrad import fmin_ncg
import sys
from pylstm.datasets import generate_memo_task

sys.path.append('.')
sys.path.append('..')

from pylstm.error_functions import MeanSquaredError
from pylstm import wrapper

rnd = np.random.RandomState(92384792)


def print_error_per_epoch(epoch, error):
    print("Epoch %d:\tTrainingerror = %0.4f" % (epoch, error))


def minibatch_generator(X, T, M, batch_size=10, loop=False):
    i = 0
    if M is None:
        M = np.ones_like(T)
    total_batches = X.shape[1]

    while True:
        if i >= total_batches:
            if not loop:
                break
            i = 0

        j = min(i + batch_size, total_batches)
        yield X[:, i:j, :], T[:, i:j, :], M[:, i:j, :]
        i += batch_size


class SgdTrainer(object):
    def __init__(self, learning_rate=0.1, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

    def train(self, net, X, T, M=None, X_val=None, T_val=None, M_val=None,
              epochs=100,
              datagenerator=None,
              minibatch_size=10,
              callback=print_error_per_epoch,
              success_criterion=lambda x: False):
        velocity = np.zeros(net.get_param_size())
        old_val_error = float('inf')
        for epoch in range(0, epochs):
            if datagenerator is not None:
                X, T, M = datagenerator()

            errors = []
            for x, t, m in minibatch_generator(X, T, M, minibatch_size):
                velocity *= self.momentum
                if self.nesterov:
                    net.param_buffer += velocity
                net.forward_pass(x)
                errors.append(net.calculate_error(t, m))
                net.backward_pass(t, m)
                dv = self.learning_rate * net.calc_gradient().flatten()
                velocity -= dv
                if self.nesterov:
                    net.param_buffer -= dv
                else:
                    net.param_buffer += velocity
                print('.', end="")
            print("")
            error = np.mean(errors)
            callback(epoch, error)
            if success_criterion(net):
                return error
            if X_val is not None:
                val_errors = []
                for x, t, m in minibatch_generator(X_val, T_val, M_val,
                                                   minibatch_size):
                    net.forward_pass(x)
                    val_errors.append(net.calculate_error(t, m))
                val_error = np.mean(val_errors)
                print("Validation Error: %0.4f" % val_error)
                if val_error > old_val_error:
                    print("Validation Error rose! Stopping.")
                    return error
                old_val_error = val_error


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
            self.stepsize = (
                self.stepsize * (increase * 1.01 + (increase == False) * .99))

            grad[:] = self.stepsize * grad_sign + -self.stepsize * (
                grad_sign == False)
            #print("grad arr:", grad_arr)
            #print("grad:", grad)
            #print(((grad_sign==False)).flatten())
            print("weights before:", weights.flatten())
            weights += grad
            print("weights after:", weights.flatten())
            self.last_grad_sign = grad_sign.copy()


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

    def train(self, net, X, T, M=None, epochs=10, minibatch_size=32, mu=1./30,
              maxiter=20, success=lambda x: True):
        mb = minibatch_generator(X, T, M, minibatch_size, loop=True)
        lambda_ = .1

        for i in range(epochs):
            print("======= Epoch %d =====" % i)
            ## calculate the gradient
            grad = np.zeros_like(net.param_buffer.flatten())
            error = []
            for x, t, m in minibatch_generator(X, T, M, minibatch_size):
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
                np.squeeze(grad), finalDW)
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
        error = net.calculate_error(T)
        callback(0, error)

        net.backward_pass(T)
        grad = 1*net.calc_gradient()


        for epoch in range(1, epochs + 1):

            #select an input batch, and target batch

            #run forward pass, output saved in out
            net.param_buffer = weights
            out = net.forward_pass(X)

            #calculate error
            error = net.calculate_error(T)
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
                return net.hessian_pass(X, v, lambda_, mu=1.0/30.0).copy().flatten() + lambda_ * v

            xopt, allvecs = fmin_ncg(f, v, grad, fhess_p=fhess_p, maxiter=150, retall=True, disp=False)


            ## backtrack #1
            prevError = error
            idx = len(allvecs)-1
            lowError = float('Inf')
            for testW in reversed(allvecs):
                net.param_buffer = weights.copy() + testW
                out = net.forward_pass(X)
                tmpError = net.calculate_error(T)
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
                tmpError = net.calculate_error(T)
                if tmpError < lowError:
                    finalDW = .9**j * bestDW
                    lowError = tmpError

            ## Levenberg-Marquardt heuristic
            drop = 2.0 / 3.0
            boost = 1/drop
            denom = 0.5*(np.dot(finalDW, fhess_p(finalDW))) + np.dot(np.squeeze(grad), finalDW)
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
    from layers import LstmLayer, RegularLayer, RnnLayer
    from datasets import generate_5bit_memory_task

    numbatches = 1
    numtimesteps = 5
    numIn = 4
    numOut = 3


    netb = NetworkBuilder()
    netb.input(4) >> LstmLayer(3, act_func="tanhx2") >> RegularLayer(4,
                                                                     act_func="softmax") >> netb.output
    net = netb.build()
    weight = rnd.randn(net.get_param_size())
    #X, T = generate_memo_task(5, 2, 32, 100)
    X, T = generate_5bit_memory_task(25)
    net.param_buffer = weight.copy()
    trainer = CgTrainer(learning_rate=0.01)
    #trainer = SgdTrainer(learning_rate=0.01)
    # X = rnd.randn(numtimesteps, numbatches,  numIn)
    # T = rnd.randn(numtimesteps, numbatches, numOut)
    trainer.train(net, X, T, epochs=100)
