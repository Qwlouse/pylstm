#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
from netbuilder import NetworkBuilder
from layers import LstmLayer
import wrapper as pw

# Instantiate a NetworkBuilder
netb = NetworkBuilder()
# add one layer of three LSTM nodes
netb.input(5) >> LstmLayer(7) >> LstmLayer(11) >> LstmLayer(3) >> netb.output
# build the network (no buffers are constructed so far)
net = netb.build()
# create some random weights (we don't care about dimensions. Just for the size)
weights = pw.BufferView(np.random.randn(net.get_param_size()))
# and set them as the parameter buffer
net.set_param_buffer(weights)
# create some random inputs (1 time slice, 1 batch, 5 features)
X = np.random.randn(2, 3, 5)
# do one forward pass (now the buffers are constructed)
out = net.forward_pass(X)
# do one backward pass (now the error buffers are constructed)
E = np.random.randn(2, 3, 3)
out_delta = net.backward_pass(E)
# the out buffer contains the results. Print them:
out.print_me()
out_delta.print_me()

# we could also access the results like this:
print("Output:")
print(out.as_array())
#print(out[0], out[1], out[2])


############ Training Example ##############################
# create and randomly initialize a network
netb = NetworkBuilder()
netb.input(5) >> LstmLayer(3) >> netb.output
net = netb.build()
net.set_param_buffer(pw.BufferView(np.random.randn(net.get_param_size())))

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