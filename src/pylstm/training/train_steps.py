#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np


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


class SgdStep(object):
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