#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals

import numpy as np

from pylstm import build_net, InputLayer, ForwardLayer
from pylstm.utils import check_gn_pass, estimate_jacobian


rnd = np.random.RandomState(2893749283)

in_size = 2
out_size = 2
nr_batches = 32
nr_timesteps = 4

net = build_net(InputLayer(in_size) >> ForwardLayer(out_size, act_func='linear'))


net.param_buffer = rnd.randn(net.get_param_size())
X = rnd.randn(nr_timesteps, nr_batches, in_size)

J = estimate_jacobian(net, X, rnd=rnd)

G = J.T.dot(J)

print(G)
v = np.arange(net.get_param_size())

o = net.hessian_pass(X, v).flatten()
p = G.dot(v).flatten()

print('hessian_pass')
print(o)
print('jacobi approx')
print(p)
print('difference')
print(o - p)

print('Sum of squares error', np.sum((o-p)**2))
print('\n\n')

print(check_gn_pass(net, X, v))




# print('\n\n')
# net.param_buffer = o-p
# for l in net.layers.keys()[1:-1]:
#     print(l)
#     w = net.get_param_view_for(l)
#     for v in w.keys():
#         print("  ", v)
#         print(w[v])
#



