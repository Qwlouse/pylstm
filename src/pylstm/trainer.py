#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import wrapper
rnd = np.random.RandomState(12345)


class MeanSquaredError(object):
    def forward_pass(self, Y, T):
        return 0.5 * np.sum((Y - T) ** 2)

    def backward_pass(self, Y, T):
        return Y - T


def print_error_per_epoch(epoch, error):
    print("Epoch %d:\tError = %0.4f" % (epoch, error))


class SgdTrainer(object):
    def __init__(self, learning_rate=0.1, error_fkt=MeanSquaredError):
        self.learning_rate = learning_rate
        self.error_fkt = error_fkt()

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
            grad_arr *= - self.learning_rate
            wrapper.add_into_b(grad, weights)

class CgTrainer(object):
    def __init__(self, learning_rate=0.1, error_fkt=MeanSquaredError):
        self.learning_rate = learning_rate
        self.error_fkt = error_fkt()

    def train(self, net, X, T, epochs=100, callback=print_error_per_epoch):
        weights = net.get_param_buffer()
        for epoch in range(1, epochs + 1):
            
            #select an input batch, and target batch
            
            #run forward pass, output saved in out
            out = net.forward_pass(X).as_array()
            
            #calculate error
            error = self.error_fkt.forward_pass(out, T) / X.shape[1]
            callback(epoch, error)
            
            #run backwards pass / calc gradient
            deltas = self.error_fkt.backward_pass(out, T)
            net.backward_pass(deltas)
            grad = net.calc_gradient()

            #initialize v, but maybe we should use the small random numbers like in old version
            v = np.zeros(net.get_param_size())

            #run cg
            #dws = cg(v, grad, lambda, mu)
            
            #but can we do this backwards
            for dwvec in dws:
                tmp_weights = weights.copy() + dwvec
                net.set_param_buffer(tmp_weight.copy())
                tmp_out = net.forward_pass(X).as_array()
                tmp_error = self.error_fkt.forward_pass(out,T) / X.shape[1]
                
                if last_error > tmp_error:
                    if n < dws.length() - 1:
                        dw = dws[n + 1]
                        break;
    
                track_last_error = track_new_error;

            #Calculate rho based on dw
            tmp_weights = weights.copy() + dw
            net.set_param_buffer(tmp_weight.copy())
            tmp_out = net.forward_pass(X).as_array()
            new_error = self.error_fkt.forward_pass(out,T) / X.shape[1]

            #f_val = cg.f_val?! 
            
            rho = (new_error - last_error)/f_val
            if rho > .75:
                lambda_ *= 2/3
            elif rho < .25: 
                lambda_ *= 3/2
            
            
            #run backtrack 2 on dw

            #grad_arr = grad.as_array()
            #grad_arr *= - self.learning_rate
            #wrapper.add_into_b(grad, weights)



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
