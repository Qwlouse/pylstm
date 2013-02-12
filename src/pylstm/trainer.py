#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import wrapper
rnd = np.random.RandomState(12345)


class MeanSquaredError(object):
    def forward_pass(self, X, T):
        return np.sum((X - T) ** 2)

    def backward_pass(self, X, T):
        return X - T


class SgdTrainer(object):
    def __init__(self, learning_rate=0.1, error_fkt=MeanSquaredError):
        self.learning_rate = learning_rate
        self.error_fkt = error_fkt()

    def train(self, net, X, T, epochs=100):
        weights = net.get_param_buffer()
        for epoch in range(1, epochs + 1):
            out = net.forward_pass(X).as_array()
            error = self.error_fkt.forward_pass(out, T)
            print("Epoch %d:\tError = %0.4f" % (epoch, error))
            deltas = self.error_fkt.backward_pass(out, T)
            net.backward_pass(deltas)
            grad = net.calc_gradient()
            grad_arr = grad.as_array()
            grad_arr *= - self.learning_rate
            wrapper.add_into_b(grad, weights)


if __name__ == "__main__":
    from netbuilder import NetworkBuilder
    from layers import LstmLayer
    netb = NetworkBuilder()
    netb.input(4) >> LstmLayer(3) >> netb.output
    net = netb.build()
    weight = rnd.randn(net.get_param_size())
    net.set_param_buffer(weight.copy())
    trainer = SgdTrainer(learning_rate=0.01)
    X = rnd.randn(2, 5, 4)
    T = rnd.randn(2, 5, 3)
    trainer.train(net, X, T, epochs=10)
