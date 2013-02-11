#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import sys
sys.path.append('.')
sys.path.append('..')
from pylstm.layers import NpFwdLayer, LstmLayer
from pylstm.netbuilder import NetworkBuilder
from scipy.optimize import approx_fprime
import numpy as np

rnd = np.random.RandomState(12345)

def check_deltas(net, X = None):
    def f(x):
        net.clear_internal_state()
        out = net.forward_pass(x.reshape(1,1,-1)).as_array()

        return .5*np.sum(out**2)

    if X is None:
        X = rnd.randn(net.get_input_size())

    delta_approx = approx_fprime(X, f, 1e-7)
    out = net.forward_pass(X.reshape(2,1,-1)).as_array()
    delta_calc = net.backward_pass(out).as_array()
    return np.sum((delta_approx - delta_calc)**2), delta_calc, delta_approx


def check_gradient(net, X = None):
    
    timesteps = 2
    X = np.ones(net.get_input_size()*timesteps)
    #X = rnd.randn(net.get_input_size()*1)

    def f(w):

        net.clear_internal_state()
        net.set_param_buffer(w)
        out = net.forward_pass(X.reshape(timesteps,1,-1)).as_array()
        
        return .5*np.sum(out**2)

    weights = np.ones(net.get_param_size())
    #weights = rnd.randn(net.get_param_size())
    #print(weights)


    #print(tmpweights)

    
    net.clear_internal_state()
    net.set_param_buffer(weights)
    out = net.forward_pass(X.reshape(timesteps,1,-1)).as_array()
    realerror = .5*(out.copy())**2
    
    differrors=np.zeros((weights.shape))
    i = 0 
    eps = 1e-7
    for x in weights:
        tmpweights = weights
        tmpweights[i] += eps        
        net.clear_internal_state()
        net.set_param_buffer(tmpweights)
        newout = net.forward_pass(X.reshape(timesteps,1,-1)).as_array()
        
    #print(newout)
        err2 = .5*(newout.copy())**2
        #print("error1: ", err1, "type: ", type(err1), " ",err1.shape)
        #print("error2: ", err2, "type: ", type(err2), " ",err2.shape) 
        
        #differrors[i] = (realerror-err2)/eps
       
        tmpweights[i] -= eps        

        i+=1
    #print(err1-err2)
    
    print(differrors)

    grad_approx = approx_fprime(weights, f, 1e-7)
    
    net.clear_internal_state()
    net.set_param_buffer(weights)
    out = net.forward_pass(X.reshape(timesteps,1,-1)).as_array()
    #print(out)
    delta_calc = net.backward_pass(out).as_array()
    grad_calc = net.calc_gradient().as_array()
    
    return np.sum((grad_approx - grad_calc.squeeze())**2), grad_calc, grad_approx



if __name__ == "__main__":
    netb  = NetworkBuilder()
    netb.input(1) >> LstmLayer(1) >> netb.output
    net = netb.build()

    #weights = rnd.randn(net.get_param_size())
    # and set them as the parameter buffer
    #net.set_param_buffer(weights)

    a, b, c = check_gradient(net)

    #print(a)
    #print(b.squeeze())
    print(c)
    print(b.squeeze())




