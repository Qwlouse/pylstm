#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


################## Callback Functions ######################
def print_error_per_epoch(epoch, net, error):
    print("Epoch %d:\tTraining error = %0.4f" % (epoch, error))


class SaveWeightsPerEpoch(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, epoch, net, error):
        np.save(self.filename, net.param_buffer)


################## Minibatch Iterators ######################
def Undivided(X, T, M):
    yield X, T, M


class Minibatches(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def __call__(self, X, T, M=None):
        i = 0
        total_batches = X.shape[1]
        while i < total_batches:
            j = min(i + self.batch_size, total_batches)
            x = X[:, i:j, :]
            t = T[i:j] if isinstance(T, list) else T[:, i:j, :]
            m = None if M is None else M[:, i:j, :]
            yield x, t, m
            i += self.batch_size


def Online(X, T, M=None):
    for i in range(X.shape[1]):
        x = X[:, i:i+1, :]
        t = T[i:i+1] if isinstance(T, list) else T[:, i:i+1, :]
        m = None
        if M is not None:
            m = M[:, i:i+1, :]
            for k in range(m.shape[0] - 1, -1, -1):
                if m[k, 0, 0] != 0:
                    x = x[:k + 1, :, :]
                    t = t[:k + 1, :, :] if not isinstance(t, list) else t
                    m = t[:k + 1, :, :]
                    break
        yield x, t, m


################## Success Criteria ######################
class ValidationErrorRises(object):
    def __init__(self, X, T, M=None, process_data=Undivided):
        self.old_val_error = float('inf')
        self.X = X
        self.T = T
        self.M = M
        self.process_data = process_data

    def restart(self):
        self.old_val_error = float('inf')

    def __call__(self, net, train_error):
        val_errors = []
        for x, t, m in self.process_data(self.X, self.T, self.M):
            net.forward_pass(x)
            val_errors.append(net.calculate_error(t, m))
        val_error = np.mean(val_errors)
        print("Validation Error: %0.4f" % val_error)
        if val_error > self.old_val_error:
            print("Validation Error rose! Stopping.")
            return True
        self.old_val_error = val_error
        return False


################## Trainer Cores ######################
class DiagnosticStep(object):
    def __init__(self):
        self.net = None

    def start(self, net):
        print("start DiagnosticStep with net=", net)

    def run(self, x, t, m):
        print("DiagnosticStep: x.shape=", x.shape)
        if isinstance(t, list):
            print("DiagnosticStep: len(t)=", len(t))
        else:
            print("DiagnosticStep: t.shape=", t.shape)
        print("DiagnosticStep: m=", m)
        return 15


class ForwardStep(object):
    def __init__(self):
        self.net = None

    def start(self, net):
        self.net = net

    def run(self, x, t, m):
        self.net.forward_pass(x)
        return self.net.calculate_error(t, m)


class SGDStep(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.net = None

    def start(self, net):
        self.net = net

    def run(self, x, t, m):
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        self.net.param_buffer -= self.learning_rate * self.net.calc_gradient().flatten()
        return error


class MomentumStep(object):
    def __init__(self, learning_rate=0.1, momentum=0.0):
        self.velocity = None
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.net = None

    def start(self, net):
        self.net = net
        self.velocity = np.zeros(net.get_param_size())

    def run(self, x, t, m):
        self.velocity *= self.momentum
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        dv = self.learning_rate * self.net.calc_gradient().flatten()
        self.velocity -= dv
        self.net.param_buffer += self.velocity
        return error


class NesterovStep(object):
    def __init__(self, learning_rate=0.1, momentum=0.0):
        self.velocity = None
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.net = None

    def start(self, net):
        self.net = net
        self.velocity = np.zeros(net.get_param_size())

    def run(self, x, t, m):
        self.velocity *= self.momentum
        self.net.param_buffer += self.velocity
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        dv = self.learning_rate * self.net.calc_gradient().flatten()
        self.velocity -= dv
        self.net.param_buffer -= dv
        return error


class RPropStep(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.net = None
        self.initialized = False

    def start(self, net):
        self.net = net
        self.initialized = False

    def run(self, x, t, m):
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        grad = self.net.calc_gradient()

        #calculate grad sign
        grad_sign = (grad > 0.0)

        if not self.initialized:
            self.last_grad_sign = grad_sign
            self.stepsize = np.ones_like(grad_sign)
            self.initialized = True
            return error

        increase = (grad_sign == self.last_grad_sign)
        self.stepsize = (
            self.stepsize * (increase * 1.01 + (increase == False) * .99))

        grad[:] = self.stepsize * grad_sign + -self.stepsize * (
            grad_sign == False)

        self.net.param_buffer -= grad * self.learning_rate

        self.last_grad_sign = grad_sign.copy()


################## Basic Trainer ######################
class Trainer(object):
    def __init__(self, net, core=None, **kwargs):
        self.net = net
        self.stepper = core if core else SGDStep(**kwargs)
        self.success_criteria = []
        self.callbacks = [print_error_per_epoch]

    def emit_callbacks(self, epoch, train_error):
        for cb in self.callbacks:
            cb(epoch, self.net, train_error)

    def is_successful(self, train_error):
        for sc in self.success_criteria:
            if sc(self.net, train_error):
                return True
        return False

    def restart_success_criteria(self):
        for sc in self.success_criteria:
            try:
                sc.restart()
            except AttributeError:
                pass

    def train(self, X, T, M=None,
              max_epochs=100,
              process_data=Undivided):
        self.stepper.start(self.net)
        self.restart_success_criteria()

        for epoch in range(1, max_epochs + 1):
            errors = []
            for x, t, m in process_data(X, T, M):
                errors.append(self.stepper.run(x, t, m))

            train_error = np.mean(errors)
            self.emit_callbacks(epoch, train_error)

            if self.is_successful(train_error):
                return train_error



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

    def train(self, net, X, T, M=None, epochs=10, minibatch_size=32, mu=1. / 30,
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


if __name__ == "__main__":
    from pylstm.netbuilder import NetworkBuilder
    from pylstm.layers import RegularLayer
    from pylstm.error_functions import CTC

    rnd = np.random.RandomState(145)

    netb = NetworkBuilder()
    netb.input(3) >> RegularLayer(6, act_func="softmax") >> netb.output
    netb.error_func = CTC
    net = netb.build()
    net.param_buffer = rnd.randn(net.get_param_size())
    X = rnd.randn(7, 5, 3)
    T = [np.array([i]*2) for i in range(1, 6)]
    M = np.ones((7, 5, 1))
    M[6, :, :] = 0
    M[5, 0, :] = 0

    trainer = Trainer(net, DiagnosticStep())
    trainer.success_criteria.append(ValidationErrorRises(X, T))

    trainer.train(X, T, M, max_epochs=50, process_data=Online)
