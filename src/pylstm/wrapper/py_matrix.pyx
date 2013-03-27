import numpy as np
cimport numpy as np
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# http://stackoverflow.com/questions/3046305
# http://article.gmane.org/gmane.comp.python.cython.user/5625

cdef class Buffer:
    def __cinit__(self, a=None, int batches=1, int features=1):
        cdef np.ndarray[np.double_t, ndim=3, mode='c'] A
        if a is None:
            self.A = None
            self.view = cm.Matrix()
        elif isinstance(a, int):
            self.A = None
            self.view = cm.Matrix(features, batches, a)
        else: # a is np array or iterable
            if len(a) == 0:
                self.view = cm.Matrix()
                self.A = None
                return
            a = np.array(a)
            if len(a.shape) == 1:
                a = a.reshape(-1, 1, 1)
            A = np.ascontiguousarray(a, dtype=np.float64)
            self.view = cm.Matrix(&A[0,0,0], A.shape[2], A.shape[1], A.shape[0])
            self.A = A # make sure numpy array does not get GCed

    def __len__(self):
        return self.view.size

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.view[item]
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or self.get_time_size()

            b = Buffer()
            b.view = self.view.subslice(start, self.view.n_rows, self.view.n_columns, stop-start)
            b.A = self.A
            return b
            
    def __setitem__(self, int item, float value):
        self.view[item] = value
        
    # from here: https://gist.github.com/1249305
    def as_array(self):
        cdef np.npy_intp shape[3]
        cdef np.ndarray ndarray
        shape[0] = <np.npy_intp> self.get_time_size()
        shape[1] = <np.npy_intp> self.get_batch_size()
        shape[2] = <np.npy_intp> self.get_feature_size()
        # Create a 3D array
        ndarray = np.PyArray_SimpleNewFromData(3, shape, np.NPY_FLOAT64, self.view.get_data())
        # Assign our object to the 'base' of the ndarray object
        ndarray.base = <PyObject*> self
        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(self)
        return ndarray

    def get_feature_size(self):
        return self.view.n_rows

    def get_batch_size(self):
        return self.view.n_columns

    def get_time_size(self):
        return self.view.n_slices

    def shape(self):
        return self.get_time_size(), self.get_batch_size(), self.get_feature_size()

    def reshape(self, time_size, batch_size, feature_size):
        if time_size == -1:
            assert batch_size >= 1
            assert feature_size >= 1
            time_size = len(self) // (batch_size * feature_size)
        elif batch_size == -1:
            assert time_size >= 1
            assert feature_size >= 1
            batch_size = len(self) // (time_size * feature_size)
        elif feature_size == -1:
            assert time_size >= 1
            assert batch_size >= 1
            feature_size = len(self) // (time_size * batch_size)
        #assert time_size >= 1
        #assert batch_size >= 1
        #assert feature_size >= 1
        assert time_size * batch_size * feature_size == len(self)
        b = Buffer()
        b.A = self.A
        b.view = cm.Matrix(self.view)
        b.view.n_rows = feature_size
        b.view.n_columns = batch_size
        b.view.n_slices = time_size 
        return b

    def print_me(self):
        self.view.print_me()

    def __repr__(self):
        t, b, f = self.shape()
        return "<Buffer (%d, %d, %d) at %#x>"%(t, b, f, id(self))


def dot(Buffer a not None, Buffer b not None, Buffer out not None):
    cm.dot(a.view, b.view, out.view)

#def add(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
#    cm.add(a.view, b.view, out.view)

def add_into_b(Buffer a not None, Buffer b not None):
    cm.add_into_b(a.view, b.view)

def add_scalar(Buffer a not None, double b):
    cm.add_scalar(a.view, b)

def mult(Buffer a not None, Buffer b not None, Buffer out not None):
    cm.mult(a.view, b.view, out.view)

def mult_add(Buffer a not None, Buffer b not None, Buffer out not None):
    cm.mult_add(a.view, b.view, out.view)

def dot(Buffer a not None, Buffer b not None, Buffer out not None):
    cm.dot(a.view, b.view, out.view)

def dot_add(Buffer a not None, Buffer b not None, Buffer out not None):
    cm.dot_add(a.view, b.view, out.view)

def apply_sigmoid(Buffer a not None, Buffer out not None):
    cm.add_into_b(a.view, out.view)

def apply_tanh(Buffer a not None, Buffer out not None):
    cm.add_into_b(a.view, out.view)

def apply_tanhx2(Buffer a not None, Buffer out not None):
    cm.add_into_b(a.view, out.view)

def equals(Buffer a not None, Buffer out not None):
    cm.add_into_b(a.view, out.view)
