#!/usr/bin/python
# coding=utf-8

from __future__ import division, print_function, unicode_literals
from pylstm.netbuilder import NetworkBuilder
from pylstm.layers import *


netb = NetworkBuilder()

l = ReverseLayer(6)

netb.input(4) >> ReverseLayer(4) >> l >> netb.output
netb.input() >> RegularLayer(2) >> l

net = netb.build()


print(net)

X = np.ones((3, 2, 4))
print(net.forward_pass(X).as_array())
print(net.backward_pass(np.ones((3, 2, 6))).as_array())
