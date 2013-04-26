#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from pylstm import NetworkBuilder, RegularLayer, RnnLayer, LstmLayer, ReverseLayer
import numpy as np
rnd = np.random.RandomState()

in_size = 3
out_size = 2
nr_batches = 1
nr_timesteps = 4

netb = NetworkBuilder()
netb.input(in_size) >> LstmLayer(out_size, act_func='linear') >> netb.output
#netb.input(in_size) >>  RegularLayer(out_size, act_func='linear') >> netb.output
net = netb.build()


net.param_buffer = rnd.randn(net.get_param_size())
X = rnd.randn(nr_timesteps, nr_batches, in_size)

out = net.forward_pass(X).copy()

J = np.zeros((out_size * nr_timesteps, net.get_param_size()))

for i in range(out_size * nr_timesteps):
    out_masked = np.zeros_like(out).flatten()
    out_masked[i] = 1.0  # out[:, :, i]
    out_masked = out_masked.reshape(*out.shape)

    deltas = net.pure_backpass(out_masked)
    J[i, :] = net.calc_gradient().flatten()

G = J.T.dot(J)
print(G.shape)

v = np.arange(net.get_param_size())

o = net.hessian_pass(X, v).flatten()
p = G.dot(v).flatten()

#print(o)
#print(p)
print(o - p)
print(np.sum(np.abs(o-p)))


print('\n\n')
net.param_buffer = o-p
for l in net.layers.keys()[1:-1]:
    print(l)
    w = net.get_param_view_for(l)
    for v in w.keys():
        print("  ", v)
        print(w[v])




