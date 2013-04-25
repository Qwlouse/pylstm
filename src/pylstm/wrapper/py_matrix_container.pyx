#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from py_matrix cimport Matrix


cdef class MatrixContainer:
    def __cinit__(self):
        self.this_ptr = NULL

    def __dealloc__(self):
        del self.this_ptr

    def __getattr__(self, item):
        if self.this_ptr.contains(item):
            b = Matrix()
            b.c_obj = self.this_ptr[0][item]
            b.A = None
            return b.as_array()
        else:
            raise AttributeError("'%s' is not a valid view."%item)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __contains__(self, item):
        return self.this_ptr.contains(item)

    def __unicode__(self):
        return "<" + self.this_ptr.get_typename() + ": " + ", ".join(self.keys()) + ">"

    def __repr__(self):
        return self.__unicode__()

    def __len__(self):
        return self.this_ptr.get_size()

    def keys(self):
        return self.this_ptr.get_view_names()

    def items(self):
        return [(n, self[n]) for n in self.keys()]

    def values(self):
        return [self[n] for n in self.keys()]


