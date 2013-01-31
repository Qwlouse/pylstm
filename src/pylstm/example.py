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
netb.input(5) >> LstmLayer(3) >> netb.output
# build the network (no buffers are constructed so far)
net = netb.build()
# create some random weights (we don't care about dimensions. Just for the size)
weights = pw.BufferView(np.random.randn(net.get_param_size()))
# and set them as the parameter buffer
net.set_param_buffer(weights)
# create some random inputs (1 time slice, 1 batch, 5 features)
X = pw.BufferView(np.random.randn(2, 3, 5))
# do one forward pass (now the buffers are constructed with t=1 and b=1)
out = net.forward_pass(X)
# the out buffer contains the results. Print them:
out.print_me()

# we could also access the results like this:
print("Output:")
print(out.as_array())
#print(out[0], out[1], out[2])