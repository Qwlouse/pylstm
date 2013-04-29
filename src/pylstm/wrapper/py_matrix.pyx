import numpy as np
cimport numpy as np
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# http://stackoverflow.com/questions/3046305
# http://article.gmane.org/gmane.comp.python.cython.user/5625

cdef class Matrix:
    def __cinit__(self, a=None, int batches=1, int features=1):
        cdef np.ndarray[np.double_t, ndim=3, mode='c'] A
        if a is None:
            self.A = None
            self.c_obj = cm.Matrix()
        elif isinstance(a, int):
            self.A = None
            self.c_obj = cm.Matrix(features, batches, a)
        else: # a is np array or iterable
            if len(a) == 0:
                self.c_obj = cm.Matrix()
                self.A = None
                return
            a = np.array(a)
            if len(a.shape) == 1:
                a = a.reshape(-1, 1, 1)
            A = np.ascontiguousarray(a, dtype=np.float64)
            self.c_obj = cm.Matrix(&A[0,0,0], A.shape[2], A.shape[1], A.shape[0])
            self.A = A # make sure numpy array does not get GCed

    def __len__(self):
        return self.c_obj.size

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.c_obj[item]
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or self.get_time_size()

            b = Matrix()
            b.c_obj = self.c_obj.sub_matrix(start, self.c_obj.n_rows, self.c_obj.n_columns, stop-start)
            b.A = self.A
            return b
            
    def __setitem__(self, int item, float value):
        self.c_obj[item] = value

    def feature_slice(self, start, stop=None):
        b = Matrix()
        b.A = self.A
        assert 0 <= start
        if self.get_feature_size() == 0:
            assert start == 0
            assert stop is None or stop == 0
            b.c_obj = self.c_obj
        else :
            assert start < self.get_feature_size()
            if stop is None:
                b.c_obj = self.c_obj.row_slice(start)
            else:
                assert start <= stop <= self.get_feature_size()
                if start == stop:
                    b.c_obj = self.c_obj.sub_matrix(0, 0, 0, 0)
                else:
                    b.c_obj = self.c_obj.row_slice(start, stop)
        return b

    def time_slice(self, start, stop=None):
        b = Matrix()
        b.A = self.A
        assert 0 <= start
        if self.get_time_size() == 0:
            assert start == 0
            assert stop is None or stop == 0
            b.c_obj = self.c_obj
        else :
            assert start < self.get_time_size()
            if stop is None:
                b.c_obj = self.c_obj.slice(start)
            else:
                assert start <= stop <= self.get_time_size()
                b.c_obj = self.c_obj.slice(start, stop)
        return b
        
    # from here: https://gist.github.com/1249305
    def as_array(self):
        cdef np.npy_intp shape[3]
        cdef np.ndarray ndarray
        shape[0] = <np.npy_intp> self.get_time_size()
        shape[1] = <np.npy_intp> self.get_batch_size()
        shape[2] = <np.npy_intp> self.get_feature_size()
        # Create a 3D array
        ndarray = np.PyArray_SimpleNewFromData(3, shape, np.NPY_FLOAT64, self.c_obj.get_data())
        # Assign our object to the 'base' of the ndarray object
        ndarray.base = <PyObject*> self
        # Increment the reference count, as the above assignment was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(self)
        return ndarray

    def get_feature_size(self):
        return self.c_obj.n_rows

    def get_batch_size(self):
        return self.c_obj.n_columns

    def get_time_size(self):
        return self.c_obj.n_slices

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
        assert time_size * batch_size * feature_size == len(self)
        b = Matrix()
        b.A = self.A
        b.c_obj = cm.Matrix(self.c_obj)
        b.c_obj.n_rows = feature_size
        b.c_obj.n_columns = batch_size
        b.c_obj.n_slices = time_size
        return b

    def print_me(self):
        self.c_obj.print_me()

    def __repr__(self):
        t, b, f = self.shape()
        return "<Matrix (%d, %d, %d) at %#x>"%(t, b, f, id(self))

    def set_all_elements_to(self, value):
        self.c_obj.set_all_elements_to(value)


def dot(Matrix a not None, Matrix b not None, Matrix out not None):
    cm.dot(a.c_obj, b.c_obj, out.c_obj)

def add_into_b(Matrix a not None, Matrix b not None):
    cm.add_into_b(a.c_obj, b.c_obj)

def add_scalar(Matrix a not None, double b):
    cm.add_scalar(a.c_obj, b)

def mult(Matrix a not None, Matrix b not None, Matrix out not None):
    cm.mult(a.c_obj, b.c_obj, out.c_obj)

def mult_add(Matrix a not None, Matrix b not None, Matrix out not None):
    cm.mult_add(a.c_obj, b.c_obj, out.c_obj)

def dot(Matrix a not None, Matrix b not None, Matrix out not None):
    cm.dot(a.c_obj, b.c_obj, out.c_obj)

def dot_add(Matrix a not None, Matrix b not None, Matrix out not None):
    cm.dot_add(a.c_obj, b.c_obj, out.c_obj)

def apply_sigmoid(Matrix a not None, Matrix out not None):
    cm.add_into_b(a.c_obj, out.c_obj)

def apply_tanh(Matrix a not None, Matrix out not None):
    cm.add_into_b(a.c_obj, out.c_obj)

def apply_tanhx2(Matrix a not None, Matrix out not None):
    cm.add_into_b(a.c_obj, out.c_obj)

def equals(Matrix a not None, Matrix out not None):
    cm.add_into_b(a.c_obj, out.c_obj)
