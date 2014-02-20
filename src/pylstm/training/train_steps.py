#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.randomness import Seedable, global_rnd
from pylstm.training.schedules import get_schedule
from pylstm.training.data_iterators import Minibatches
from pylstm.targets import Targets
from pylstm.training.cg_trainer import conjgrad3


class TrainingStep(object):
    """
    Base class for all training steps. Defines the common interface
    """
    def __init__(self):
        self.net = None

    def start(self, net):
        self.net = net
        self._initialize()

    def _initialize(self):
        pass

    def run(self, x, t, m):
        pass


class DiagnosticStep(TrainingStep):
    """
    Only prints debugging information. Does not train at all.
    Use for diagnostics only.
    """
    def _initialize(self):
        print("start DiagnosticStep with net=", self.net)

    def run(self, x, t, m):
        print("DiagnosticStep: x.shape=", x.shape)
        if isinstance(t, Targets):
            print("DiagnosticStep: t=", t)
        else:
            print("DiagnosticStep: t.shape=", t.shape)
        print("DiagnosticStep: m.shape=", m.shape if m is not None else None)
        return 15


class ForwardStep(TrainingStep):
    """
    Only runs the forward pass and returns the error. Does not train the
    network at all.
    This step is usually used for validation. If this step is used during
    training it should be initialized with the use_training_pass flag set to
    true.
    """

    def __init__(self, use_training_pass=False):
        super(ForwardStep, self).__init__()
        self.use_training_pass = use_training_pass

    def run(self, x, t, m):
        self.net.forward_pass(x, self.use_training_pass)
        return self.net.calculate_error(t, m)


class SgdStep(TrainingStep):
    """
    Stochastic Gradient Descent.
    """
    def __init__(self, learning_rate=0.1):
        super(SgdStep, self).__init__()
        self.learning_rate_schedule = get_schedule(learning_rate)

    def run(self, x, t, m):
        learning_rate = self.learning_rate_schedule()
        self.net.forward_pass(x, training_pass=True)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        self.net.param_buffer -= (learning_rate *
                                  self.net.calc_gradient().flatten())
        return error


class MomentumStep(TrainingStep):
    """
    Stochastic Gradient Descent with a momentum term.
    learning_rate and momentum can be scheduled using pylstm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    def __init__(self, learning_rate=0.1, momentum=0.0, scale_learning_rate=True):
        super(MomentumStep, self).__init__()
        self.velocity = None
        self.momentum_schedule = get_schedule(momentum)
        self.learning_rate_schedule = get_schedule(learning_rate)
        assert (scale_learning_rate is True) or (scale_learning_rate is False),\
            "scale_learning_rate must be True or False"
        self.scale_learning_rate = scale_learning_rate

    def _initialize(self):
        self.velocity = np.zeros(self.net.get_param_size())

    def run(self, x, t, m):
        learning_rate = self.learning_rate_schedule()
        momentum = self.momentum_schedule()
        self.velocity *= momentum
        self.net.forward_pass(x, training_pass=True)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        if self.scale_learning_rate:
            dv = (1 - momentum) * learning_rate * self.net.calc_gradient().flatten()
        else:
            dv = learning_rate * self.net.calc_gradient().flatten()

        self.velocity -= dv
        self.net.param_buffer += self.velocity
        return error


class NesterovStep(TrainingStep):
    """
    Stochastic Gradient Descent with a Nesterov-style momentum term.
    learning_rate and momentum can be scheduled using pylstm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    def __init__(self, learning_rate=0.1, momentum=0.0, scale_learning_rate=True):
        super(NesterovStep, self).__init__()
        self.velocity = None
        self.momentum_schedule = get_schedule(momentum)
        self.learning_rate_schedule = get_schedule(learning_rate)
        assert (scale_learning_rate is True) or (scale_learning_rate is False), \
            "scale_learning_rate must be True or False"
        self.scale_learning_rate = scale_learning_rate

    def _initialize(self):
        self.velocity = np.zeros(self.net.get_param_size())

    def run(self, x, t, m):
        learning_rate = self.learning_rate_schedule()
        momentum = self.momentum_schedule()
        self.velocity *= momentum
        self.net.param_buffer += self.velocity
        self.net.forward_pass(x, training_pass=True)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        if self.scale_learning_rate:
            dv = (1 - momentum) * learning_rate * self.net.calc_gradient().flatten()
        else:
            dv = learning_rate * self.net.calc_gradient().flatten()

        self.velocity -= dv
        self.net.param_buffer -= dv
        return error


class RPropStep(TrainingStep):
    """
    References:
    Improving the Rprop Learning Algorithm. Igel and Husken (2000).
    Rprop - Description and Implementation Details. Reidmiller (1994).

    Rprop default is Rprop+ which includes backtracking (even when error drops
    and gradient changes sign)
    Rprop- can be obtained by setting backtracking = False
    iRprop+ can be obtained by setting backtracking = True but
    backtrack_on_error_drop = False
    """

    def __init__(self, eta_minus=0.5, eta_plus=1.2, delta_0=0.1, delta_min=1e-6,
                 delta_max=50, backtracking=True, backtrack_on_error_drop=True):
        super(RPropStep, self).__init__()
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta = delta_0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.initialized = False
        self.backtracking = backtracking
        self.backtrack_on_error_drop = backtrack_on_error_drop
        self.last_grad_sign = 0
        self.last_update = 0
        self.last_error = np.Inf

    def _initialize(self):
        self.last_grad_sign = 0
        self.last_update = 0
        self.last_error = np.Inf
        self.initialized = True

    def run(self, x, t, m):
        self.net.forward_pass(x, training_pass=True)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        grad = self.net.calc_gradient()

        grad_sign = np.sign(grad)
        sign_flip = grad_sign * self.last_grad_sign

        # Calculate the delta
        self.delta = ((self.eta_plus * self.delta) * (sign_flip > 0) +
                      (self.eta_minus * self.delta) * (sign_flip < 0) +
                      self.delta * (sign_flip == 0))
        self.delta = np.clip(self.delta, self.delta_min, self.delta_max)

        # Calculate the update
        if self.backtracking:
            if self.backtrack_on_error_drop:
                update = (-np.sign(grad) * self.delta * (sign_flip >= 0)) + \
                         (-self.last_update * (sign_flip < 0))
            else:
                update = (-np.sign(grad) * self.delta * (sign_flip >= 0)) + \
                         (-self.last_update * (sign_flip < 0) *
                          (error > self.last_error))
        else:
            update = -np.sign(grad) * self.delta

        # Update
        self.net.param_buffer += update.flatten()

        if self.backtracking:
            self.last_grad_sign = grad_sign.copy() * (sign_flip >= 0)
        else:
            self.last_grad_sign = grad_sign.copy()
        self.last_update = update.copy()
        self.last_error = error
        return error


class RmsPropStep(TrainingStep):
    def __init__(self, step_rate=0.1, decay=0.9, momentum=0.0, step_adapt=False,
                 step_rate_min=0, step_rate_max=np.inf):
        super(RmsPropStep, self).__init__()
        self.step_rate = step_rate
        self.decay = decay
        self.momentum = momentum
        self.step_adapt = step_adapt
        self.step_rate_min = step_rate_min
        self.step_rate_max = step_rate_max
        self.scaling_factor = None

    def _initialize(self):
        self.scaling_factor = np.zeros(self.net.get_param_size())

    def run(self, x, t, m):
        self.net.forward_pass(x, training_pass=True)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        grad = self.net.calc_gradient()

        self.scaling_factor = ((1 - self.decay) * grad**2 +
                               self.decay * self.scaling_factor)
        update = (self.step_rate / self.scaling_factor) * grad
        self.net.param_buffer += update.flatten()
        return error


def calculate_gradient(net, data_iter):
        grad = np.zeros_like(net.param_buffer.flatten())
        error = []
        for x, t, m in data_iter():
            net.forward_pass(x, training_pass=True)
            net.backward_pass(t, m)
            error.append(net.calculate_error(t, m))
            grad += net.calc_gradient().flatten()
        error = np.mean(error)
        return error, grad


class CgStep(TrainingStep, Seedable):
    def __init__(self, minibatch_size=32, mu=1. / 30, maxiter=300, seed=None, matching_loss=True):
        TrainingStep.__init__(self)
        Seedable.__init__(self, seed, category='trainer')
        self.minibatch_size = minibatch_size
        self.mu = mu
        self.lambda_ = 0.1
        self.maxiter = maxiter
        self.matching_loss = matching_loss


    def _initialize(self):
        self.lambda_ = 0.1

    def _get_random_subset(self, X, T, M, subset_size):
        subset_idx = self.rnd.choice(X.shape[1], subset_size, replace=False)
        x = X[:, subset_idx, :]
        t = T[subset_idx]
        m = M[:, subset_idx, :] if M is not None else None
        return x, t, m

    def run(self, X, T, M):
        ## calculate the gradient and initial error
        data_iter = Minibatches(X, T, M, self.minibatch_size, verbose=False,
                                shuffle=False)
        error, grad = calculate_gradient(self.net, data_iter)

        ## initialize v
        #v = np.zeros(net.get_param_size())
        try:
            v = self.new_v
        except:
            v = .000001 * self.rnd.randn(self.net.get_param_size())

        # select a random subset of the data for the CG
        x, t, m = self._get_random_subset(X, T, M, self.minibatch_size)

        ## define hessian pass
        def fhess_p(v):
            return self.net.hessian_pass(x, v, t, m, self.mu, self.lambda_, self.matching_loss).copy().\
                       flatten() + self.lambda_*v

        ## run CG
        all_v = conjgrad3(grad, v.copy(), fhess_p, maxiter=self.maxiter)
        self.new_v = all_v[-1]

        ## backtrack #1
        lowError = float('Inf')
        lowIdx = 0
        weights = self.net.param_buffer.copy()
        for i, testW in reversed(list(enumerate(all_v))):
            self.net.param_buffer = weights - testW
            self.net.forward_pass(x, training_pass=True)
            tmpError = self.net.calculate_error(t, m)
            if tmpError < lowError:
                lowError = tmpError
                lowIdx = i
        bestDW = all_v[lowIdx]

        ## backtrack #2
        finalDW = bestDW
        for j in np.arange(0, 1.0, 0.1):
            tmpDW = j * bestDW
            self.net.param_buffer = weights - tmpDW
            self.net.forward_pass(x, training_pass=True)
            tmpError = self.net.calculate_error(t, m)
            if tmpError < lowError:
                finalDW = tmpDW
                lowError = tmpError

        ## Levenberg-Marquardt heuristic
        boost = 3.0 / 2.0
        self.net.param_buffer = weights
        denom = 0.5 * (np.dot(finalDW, fhess_p(finalDW))) - np.dot(
            np.squeeze(grad), finalDW)
        rho = (lowError - error) / denom
        if rho < 0.25:
            self.lambda_ *= boost
        elif rho > 0.75:
            self.lambda_ /= boost

        ## update weights
        self.net.param_buffer = weights - finalDW

        return lowError