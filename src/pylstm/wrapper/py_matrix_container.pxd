#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
cimport c_matrix as cm

cdef class MatrixContainerSlice:
    cdef cm.MatrixContainerSlice* this_ptr

cdef class MatrixContainer:
    cdef cm.MatrixContainer* this_ptr

cdef inline create_MatrixContainer(cm.MatrixContainer* c):
    bc = MatrixContainer()
    bc.this_ptr = c
    return bc

cdef inline create_MatrixContainerSlice(cm.MatrixContainerSlice* cs):
    bc = MatrixContainerSlice()
    bc.this_ptr = cs
    return bc
