#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import LstmLayer

# Instantiate a NetworkBuilder
netb = NetworkBuilder()
# add one layer of three LSTM nodes
netb.input(5) >> LstmLayer(3) >> netb.output

# build the network (no buffers are constructed so far)
net = netb.build()
# create some random weights (we don't care about dimensions. Just for the size)
weights = np.random.randn(net.get_param_size())
# and set them as the parameter buffer
net.set_param_buffer(weights)
# create some random inputs (2 time slice, 3 batch, 5 features)
X = np.random.randn(2, 3, 5)
# do one forward pass (now the buffers are constructed)
out = net.forward_pass(X)
# do one backward pass (now the error buffers are constructed)
E = np.random.randn(2, 3, 3)
out_delta = net.backward_pass(E)
# the out buffer contains the results. Print them:
out.print_me()
print("Deltas:")
out_delta.print_me()
#grad = net.calc_gradient()
#print("Gradient:")

print("Output1:", out)
############ Accessing Weights Example ##############################
lstm_weights = net.get_param_view_for("LstmLayer")
print("LSTM Weights for IX:")
lstm_weights.get_IX().print_me()


############ Training Example ##############################
# create and randomly initialize a network
#netb = NetworkBuilder()
#netb.input(5) >> LstmLayer(3) >> netb.output
#net = netb.build()
#net.set_param_buffer(pw.BufferView(np.random.randn(net.get_param_size())))

# Create a Training Dataset
X = np.random.randn(10, 50, 5)
T = np.random.randn(10, 50, 3)

#t = SgdTrainer(learning_rate=.5, momentum=0.1, error_fkt=mse)
#t.train(net, X, T, epochs=100)

#out = net.forward_pass(X)
#deltas = T - out
#d = net.backward_pass(deltas)
#grad = np.zeros(net.get_param_size())
#net.calculate_gradient(grad)
