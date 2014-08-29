#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

from pylstm.describable import Describable
from pylstm.randomness import Seedable
from pylstm.targets import Targets
from pylstm.datasets.data_iterators import Minibatches
from .schedules import get_schedule
from .utils import conjugate_gradient


############################ Base Class ########################################

class TrainingStep(Describable):
    """
    Base class for all training steps. Defines the common interface
    """
    __undescribed__ = {'net'}

    def __init__(self):
        self.net = None

    def start(self, net):
        self.net = net
        self._initialize()

    def _initialize(self):
        pass

    def run(self, input_data, targets):
        pass


############################ Training Steps ####################################

class DiagnosticStep(TrainingStep):
    """
    Only prints debugging information. Does not train at all.
    Use for diagnostics only.
    """
    def _initialize(self):
        print("start DiagnosticStep with net=", self.net)

    def run(self, input_data, targets):
        print("DiagnosticStep: x.shape=", input_data.shape)
        assert isinstance(targets, Targets)
        print("DiagnosticStep: t=", targets)
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

    def run(self, input_data, targets):
        self.net.forward_pass(input_data, training_pass=self.use_training_pass)
        return self.net.calculate_error(targets)


class SgdStep(TrainingStep):
    """
    Stochastic Gradient Descent.
    """
    def __init__(self, learning_rate=0.1):
        super(SgdStep, self).__init__()
        self.learning_rate_schedule = get_schedule(learning_rate)

    def run(self, input_data, targets):
        learning_rate = self.learning_rate_schedule()
        self.net.forward_pass(input_data, training_pass=True)
        error = self.net.calculate_error(targets)
        self.net.backward_pass(targets)
        self.net.param_buffer -= (learning_rate *
                                  self.net.calc_gradient().flatten())
        return error

    def __init_from_description__(self, description):
        self.learning_rate_schedule = get_schedule(self.learning_rate_schedule)


class MomentumStep(TrainingStep):
    """
    Stochastic Gradient Descent with a momentum term.
    learning_rate and momentum can be scheduled using pylstm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    __undescribed__ = {'velocity'}
    __default_values__ = {'scale_learning_rate': True}

    def __init__(self, learning_rate=0.1, momentum=0.0,
                 scale_learning_rate=True):
        super(MomentumStep, self).__init__()
        self.velocity = None
        self.momentum = get_schedule(momentum)
        self.learning_rate = get_schedule(learning_rate)
        assert scale_learning_rate in (True, False), \
            "scale_learning_rate must be boolen"
        self.scale_learning_rate = scale_learning_rate

    def _initialize(self):
        self.velocity = np.zeros(self.net.get_param_size())

    def run(self, input_data, targets):
        learning_rate = self.learning_rate()
        momentum = self.momentum()
        self.velocity *= momentum
        self.net.forward_pass(input_data, training_pass=True)
        error = self.net.calculate_error(targets)
        self.net.backward_pass(targets)
        if self.scale_learning_rate:
            dv = (1 - momentum) * learning_rate * \
                self.net.calc_gradient().flatten()
        else:
            dv = learning_rate * self.net.calc_gradient().flatten()

        self.velocity -= dv
        self.net.param_buffer += self.velocity
        return error

    def __init_from_description__(self, description):
        self.learning_rate = get_schedule(self.learning_rate)
        self.momentum = get_schedule(self.momentum)


class NesterovStep(MomentumStep):
    """
    Stochastic Gradient Descent with a Nesterov-style momentum term.
    learning_rate and momentum can be scheduled using pylstm.training.schedules
    If scale_learning_rate is True (default),
    learning_rate is multiplied by (1 - momentum) when used.
    """
    def run(self, input_data, targets):
        learning_rate = self.learning_rate()
        momentum = self.momentum()
        self.velocity *= momentum
        self.net.param_buffer += self.velocity
        self.net.forward_pass(input_data, training_pass=True)
        error = self.net.calculate_error(targets)
        self.net.backward_pass(targets)
        if self.scale_learning_rate:
            dv = (1 - momentum) * learning_rate * \
                self.net.calc_gradient().flatten()
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

    __undescribed__ = {
        'initialized': False,
        'last_grad_sign': 0,
        'last_update': 0,
        'last_error': np.Inf
    }

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

    def run(self, input_data, targets):
        self.net.forward_pass(input_data, training_pass=True)
        error = self.net.calculate_error(targets)
        self.net.backward_pass(targets)
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
            self.last_grad_sign = grad_sign * (sign_flip >= 0)
        else:
            self.last_grad_sign = grad_sign
        self.last_update = update.copy()
        self.last_error = error
        return error


class RmsPropStep(TrainingStep):
    __undescribed__ = {'scaling_factor'}

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

    def run(self, input_data, targets):
        self.net.forward_pass(input_data, training_pass=True)
        error = self.net.calculate_error(targets)
        self.net.backward_pass(targets)
        grad = self.net.calc_gradient()

        self.scaling_factor = ((1 - self.decay) * grad**2 +
                               self.decay * self.scaling_factor)
        # update = (self.step_rate / self.scaling_factor) * grad
        # self.net.param_buffer += update.flatten()
        return error


def calculate_gradient(net, data_iter):
        grad = np.zeros_like(net.param_buffer.flatten())
        error = []
        for x, t in data_iter():
            net.forward_pass(x, training_pass=True)
            net.backward_pass(t)
            error.append(net.calculate_error(t))
            grad += net.calc_gradient().flatten()
        error = np.mean(error)
        return error, grad


class CgStep(Seedable, TrainingStep):
    __undescribed__ = {
        'lambda': 0.1
    }

    def __init__(self, minibatch_size=32, mu=1. / 30, maxiter=300, seed=None,
                 matching_loss=True):
        TrainingStep.__init__(self)
        Seedable.__init__(self, seed)
        self.minibatch_size = minibatch_size
        self.mu = mu
        self.lambda_ = 0.1
        self.maxiter = maxiter
        self.matching_loss = matching_loss

    def _initialize(self):
        self.lambda_ = 0.1

    def _get_random_subset(self, input_data, targets, subset_size):
        subset_idx = self.rnd.choice(input_data.shape[1], subset_size,
                                     replace=False)
        x = input_data[:, subset_idx, :]
        t = targets[subset_idx]
        return x, t

    def run(self, input_data, targets):
        ## calculate the gradient and initial error
        data_iter = Minibatches(input_data, targets, self.minibatch_size,
                                verbose=False, shuffle=False)
        error, grad = calculate_gradient(self.net, data_iter)

        ## initialize v
        v_init = np.array(.000001 * self.rnd.randn(self.net.get_param_size()))

        # select a random subset of the data for the CG
        x, t = self._get_random_subset(input_data, targets, self.minibatch_size)

        ## define hessian pass
        def fhess_p(v):
            return (self.net.hessian_pass(x, v, t, self.mu, self.lambda_,
                                          self.matching_loss).copy()
                    .flatten() + self.lambda_*v) / self.minibatch_size

        ## run CG
        all_v = conjugate_gradient(grad, v_init.copy(), fhess_p,
                                   maxiter=self.maxiter)

        ## backtrack #1
        low_error = float('Inf')
        low_idx = 0
        weights = self.net.param_buffer.copy()
        for i, testW in reversed(list(enumerate(all_v))):
            self.net.param_buffer = weights - testW
            self.net.forward_pass(x, training_pass=True)
            tmp_error = self.net.calculate_error(t)
            if tmp_error < low_error:
                low_error = tmp_error
                low_idx = i
        best_dw = all_v[low_idx]

        ## backtrack #2
        final_dw = best_dw
        for j in np.arange(0, 1.0, 0.02):
            tmp_dw = j * best_dw
            self.net.param_buffer = weights - tmp_dw
            self.net.forward_pass(x, training_pass=True)
            tmp_error = self.net.calculate_error(t)
            if tmp_error < low_error:
                final_dw = tmp_dw
                low_error = tmp_error

        ## Levenberg-Marquardt heuristic
        boost = 3.0 / 2.0
        self.net.param_buffer = weights
        denom = 0.5 * (np.dot(final_dw, fhess_p(final_dw))) - np.dot(
            np.squeeze(grad), final_dw)
        rho = (low_error - error) / denom
        if rho < 0.25:
            self.lambda_ *= boost
        elif rho > 0.75:
            self.lambda_ /= boost

        ## update weights
        self.net.param_buffer = weights - final_dw

        return low_error