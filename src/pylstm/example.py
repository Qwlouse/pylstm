#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
from pylstm.trainer import SgdTrainer, RPropTrainer

from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import LstmLayer, RegularLayer
from datasets import generate_memo_problem

# # Instantiate a NetworkBuilder
# netb = NetworkBuilder()
# # add one layer of three LSTM nodes
# netb.input(2) >> LstmLayer(3) >> netb.output
# # build the network (no buffers are constructed so far)
# net = netb.build()
#
# # create some random weights (we don't care about dimensions. Just for the size)
# weights = np.ones(net.get_param_size())
# net.set_param_buffer(weights)  # and set them as the parameter buffer
# # create some random inputs (2 time slice, 3 batch, 5 features)
# # X = np.random.randn(2, 3, 5)
# X = np.ones((2, 3, 2))
#
# # do one forward pass (now the buffers are constructed)
# out = net.forward_pass(X)
# print("Output:")
# out.print_me()
# # do one backward pass (now the error buffers are constructed)
# E = np.random.randn(2, 3, 3)
# out_delta = net.backward_pass(E)
#
# # the out buffer contains the results. Print them:
# print("Deltas:")
# out_delta.print_me()
#grad = net.calc_gradient()
#print("Gradient:")


############ Accessing Weights Example ##############################
# lstm_weights = net.get_param_view_for("LstmLayer")
# print("LSTM Weights for IX:")
# lstm_weights.get_IX().print_me()

# ############ Training Example ##############################
# create and randomly initialize a network
netb = NetworkBuilder()
netb.input(2) >> LstmLayer(2) >> netb.output
net = netb.build()
net.set_param_buffer(np.random.randn(net.get_param_size()))

# Generate 5bit problem
timesteps = 30
X, T = generate_memo_problem(5,  2, 32, timesteps)

t = SgdTrainer(learning_rate=.01)
t.train(net, X, T, epochs=50)

# t = RPropTrainer(learning_rate=.01)
# t.train(net, X, T, epochs=2)

# ############ Complex Architecture Example ##############################
# create and randomly initialize a network
# netb = NetworkBuilder()
# l = FwdLayer(2)
# netb.input(2) >> LstmLayer(5) >> l >> netb.output
# netb.input() >> LstmLayer(5) >> l
# net = netb.build()
# net.set_param_buffer(np.random.randn(net.get_param_size()))
#
# t.train(net, X, T, epochs=50)
