#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
import time


################## Callback Functions ######################
def print_error_per_epoch(epoch, net, training_errors, validation_errors):
    if len(validation_errors) == 0:
        print("\nEpoch %d:\tTraining error = %0.4f" % (epoch,
                                                       training_errors[-1]))
    else:
        print("\nEpoch %d:\tTraining error = %0.4f Validation error = %0.4f" %
              (epoch, training_errors[-1], validation_errors[-1]))


class SaveWeightsPerEpoch(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, epoch, net, training_errors, validation_errors):
        np.save(self.filename.format(epoch=epoch), net.param_buffer)


class SaveBestWeights(object):
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, epoch, net, training_errors, validation_errors):
        e = validation_errors if len(validation_errors) > 0 else training_errors
        if np.argmin(e) == len(e) - 1:
            filename = self.filename.format(epoch=epoch)
            print("Saving weights to {0}...".format(filename))
            np.save(filename, net.param_buffer)


################## Dataset Iterators ######################
class Undivided(object):
    def __init__(self, X, T, M=None):
        self.X = X
        self.T = T
        self.M = M

    def __call__(self):
        yield self.X, self.T, self.M


class Minibatches(object):
    def __init__(self, X, T, M=None, batch_size=1):
        self.X = X
        self.T = T
        self.M = M
        self.batch_size = batch_size

    def __call__(self):
        i = 0
        update_progress(0)
        total_batches = self.X.shape[1]
        while i < total_batches:
            j = min(i + self.batch_size, total_batches)
            x = self.X[:, i:j, :]
            t = self.T[i:j] if isinstance(self.T, list) else self.T[:, i:j, :]
            m = None if self.M is None else self.M[:, i:j, :]
            yield x, t, m
            i += self.batch_size
            update_progress(i/total_batches)


class Online(object):
    def __init__(self, X, T, M=None, verbose=True):
        self.X = X
        self.T = T
        self.M = M
        self.verbose = verbose

    def __call__(self):
        i = 0
        if self.verbose:
            update_progress(0)
        total_batches = self.X.shape[1]
        while i < total_batches:
            x = self.X[:, i:i+1, :]
            t = self.T[i:i+1] if isinstance(self.T, list) else \
                self.T[:, i:i+1, :]
            m = None
            if self.M is not None:
                m = self.M[:, i:i+1, :]
                for k in range(m.shape[0] - 1, -1, -1):
                    if m[k, 0, 0] != 0:
                        x = x[:k + 1, :, :]
                        t = t[:k + 1, :, :] if not isinstance(t, list) else t
                        m = m[:k + 1, :, :]
                        break
            yield x, t, m
            i += 1
            if self.verbose:
                update_progress(i/total_batches)


def update_progress(progress):
    barLength = 50  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress: [{0}{1}] {2}% {3}".format("#" * block,
                                                  "-" * (barLength-block),
                                                  round(progress*100, 2),
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()


################## Dataset Modifications ######################
class Noisy(object):
    def __init__(self, data_iter, std=0.1, rnd=np.random.RandomState()):
        self.f = data_iter
        self.rnd = rnd
        self.std = std

    def __call__(self):
        for x, t, m in self.f():
            x_noisy = x + self.rnd.randn(*x.shape) * self.std
            yield x_noisy, t, m


################## Success Criteria ######################
class ValidationErrorRises(object):
    def __init__(self, delay=1):
        self.delay = delay

    def restart(self):
        pass

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        best_val_error = np.argmin(validation_errors)
        if len(validation_errors) > best_val_error + self.delay:
            print("Validation error did not fall for %d epochs! Stopping."
                  % self.delay)
            return True
        return False


class MaxEpochsSeen(object):
    def __init__(self, max_epochs=100):
        self.max_epochs = max_epochs

    def restart(self):
        pass

    def __call__(self, epochs_seen, net, training_errors, validation_errors):
        if epochs_seen >= self.max_epochs:
            return True


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
        self.net.param_buffer -= self.learning_rate * \
                                 self.net.calc_gradient().flatten()
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
        self.validation_stepper = ForwardStep()
        self.success_criteria = []
        self.callbacks = [print_error_per_epoch]
        self.training_errors = []
        self.validation_errors = []
        self.epochs_seen = 0

    def emit_callbacks(self):
        for cb in self.callbacks:
            cb(self.epochs_seen, self.net,
               self.training_errors, self.validation_errors)

    def is_successful(self):
        for sc in self.success_criteria:
            if sc(self.epochs_seen, self.net,
                  self.training_errors, self.validation_errors):
                return True
        return False

    def restart_success_criteria(self):
        for sc in self.success_criteria:
            try:
                sc.restart()
            except AttributeError:
                pass

    def train(self, training_data_getter, validation_data_getter=None):
        # May add a default MaxEpochsSeen here if that feels better to the soul
        self.stepper.start(self.net)
        self.validation_stepper.start(self.net)
        self.restart_success_criteria()

        while True:
            train_errors = []
            print("\nTraining ...")
            start = time.time()
            for x, t, m in training_data_getter():
                train_errors.append(self.stepper.run(x, t, m))
            print("Wall Time taken: ", time.time() - start)

            train_error = np.mean(train_errors)
            self.training_errors.append(train_error)

            if validation_data_getter is not None:
                valid_errors = []
                print("Validating ...")
                start = time.time()
                for x, t, m in validation_data_getter():
                    valid_errors.append(self.validation_stepper.run(x, t, m))
                print("Wall Time taken: ", time.time() - start)

                valid_error = np.mean(valid_errors)
                self.validation_errors.append(valid_error)

            self.epochs_seen += 1

            self.emit_callbacks()

            if self.is_successful():
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

    def train(self, net, X, T, M=None, epochs=10, minibatch_size=32,
              mu=1. / 30, maxiter=20, success=lambda x: False):
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


if __name__ == "__main__":
    from pylstm.netbuilder import NetworkBuilder
    from pylstm.layers import ForwardLayer
    from pylstm.error_functions import MeanSquaredError

    rnd = np.random.RandomState(145)

    netb = NetworkBuilder()
    netb.input(3) >> ForwardLayer(6, act_func="softmax") >> netb.output
    netb.error_func = MeanSquaredError
    net = netb.build()
    net.param_buffer = rnd.randn(net.get_param_size())
    X = rnd.randn(7, 200, 3)
    T = rnd.randn(7, 200, 6)
    M = np.ones((7, 200, 1))
    M[6, :, :] = 0
    M[5, 0, :] = 0

    trainer = Trainer(net, SGDStep(learning_rate=0.1))
    trainer.success_criteria.append(ValidationErrorRises())

    trainer.train(training_data_getter=Noisy(std=0.05, rnd=rnd,
                                             data_iter=Online(X, T, M)),
                  validation_data_getter=Online(X, T))
