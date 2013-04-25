#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
cimport c_layers as cl


cdef class ViewContainer:
    cdef cl.ViewContainer* this_ptr

cdef inline create_ViewContainer(cl.ViewContainer* c):
    bc = ViewContainer()
    bc.this_ptr = c
    return bc