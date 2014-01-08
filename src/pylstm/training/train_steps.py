#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from pylstm.targets import Targets


class DiagnosticStep(object):
    def __init__(self):
        self.net = None

    def start(self, net):
        print("start DiagnosticStep with net=", net)

    def run(self, x, t, m):
        print("DiagnosticStep: x.shape=", x.shape)
        if isinstance(t, Targets):
            print("DiagnosticStep: t=", t)
        else:
            print("DiagnosticStep: t.shape=", t.shape)
        print("DiagnosticStep: m.shape=", m.shape if m is not None else None)
        return 15


class ForwardStep(object):
    def __init__(self):
        self.net = None

    def start(self, net):
        self.net = net

    def run(self, x, t, m):
        self.net.forward_pass(x)
        return self.net.calculate_error(t, m)


class SgdStep(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.net = None

    def start(self, net):
        self.net = net

    def run(self, x, t, m):
        learning_rate = self.learning_rate if isinstance(self.learning_rate, (int, float)) else self.learning_rate()
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        self.net.param_buffer -= learning_rate * \
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
        learning_rate = self.learning_rate if isinstance(self.learning_rate, (int, float)) else self.learning_rate()
        momentum = self.momentum if isinstance(self.momentum, (int, float)) else self.momentum()
        self.velocity *= momentum
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        dv = learning_rate * self.net.calc_gradient().flatten()
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
        learning_rate = self.learning_rate if isinstance(self.learning_rate, (int, float)) else self.learning_rate()
        momentum = self.momentum if isinstance(self.momentum, (int, float)) else self.momentum()
        self.velocity *= momentum
        self.net.param_buffer += self.velocity
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        dv = learning_rate * self.net.calc_gradient().flatten()
        self.velocity -= dv
        self.net.param_buffer -= dv
        return error


class RPropStep(object):
    """
    References:
    Improving the Rprop Learning Algorithm. Igel and Husken (2000).
    Rprop - Description and Implementation Details. Reidmiller (1994).

    Rprop default is Rprop+ which includes backtracking (even when error drops and gradient changes sign)
    Rprop- can be obtained by setting backtracking = False
    iRprop+ can be obtained by setting backtracking = True but backtrack_on_error_drop = False
    """
    def __init__(self, eta_minus=0.5, eta_plus=1.2, delta_0=0.1, delta_min=1e-6, delta_max=50,
                 backtracking=True, backtrack_on_error_drop=True):
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta = delta_0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.initialized = False
        self.backtracking = backtracking
        self.backtrack_on_error_drop = backtrack_on_error_drop

    def start(self, net):
        self.net = net
        self.last_grad_sign = 0
        self.last_update = 0
        self.last_error = np.Inf
        self.initialized = True

    def run(self, x, t, m):
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        grad = self.net.calc_gradient()

        grad_sign = np.sign(grad)
        sign_flip = grad_sign * self.last_grad_sign

        # Calculate the delta
        self.delta = (self.eta_plus * self.delta) * (sign_flip > 0) + \
                     (self.eta_minus * self.delta) * (sign_flip < 0) + \
                      self.delta * (sign_flip == 0)
        self.delta = np.clip(self.delta, self.delta_min, self.delta_max)

        # Calculate the update
        if self.backtracking:
            if self.backtrack_on_error_drop:
                update = (-np.sign(grad) * self.delta * (sign_flip >= 0)) + \
                         (-self.last_update * (sign_flip < 0))
            else:
                update = (-np.sign(grad) * self.delta * (sign_flip >= 0)) + \
                         (-self.last_update * (sign_flip < 0) * (error > self.last_error))
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


class RmsPropStep(object):
    def __init__(self, step_rate=0.1, decay=0.9, momentum=0.0,
                 step_adapt=False, step_rate_min=0, step_rate_max=np.inf):
        self.step_rate = step_rate
        self.decay = decay
        self.momentum = momentum
        self.step_adapt = step_adapt
        self.step_rate_min = step_rate_min
        self.step_rate_max = step_rate_max

    def start(self, net):
        self.net = net
        self.scaling_factor = np.zeros(net.get_param_size())

    def run(self, x, t, m):
        self.net.forward_pass(x)
        error = self.net.calculate_error(t, m)
        self.net.backward_pass(t, m)
        grad = self.net.calc_gradient()

        self.scaling_factor = (1 - self.decay) * grad**2 + self.decay * self.scaling_factor
        update = (self.step_rate / self.scaling_factor) * grad
        #self.net.param_buffer += update.flatten()

        return error
