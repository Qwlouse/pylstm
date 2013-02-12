import numpy as np
cimport numpy as np
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# http://stackoverflow.com/questions/3046305
# http://article.gmane.org/gmane.comp.python.cython.user/5625

cdef class Buffer:
    def __cinit__(self, a):
        cdef np.ndarray[np.double_t, ndim=3, mode='c'] A
        if isinstance(a, int):
            self.thisptr = new cm.MatrixCPU(1, 1, a)
        else:
            if len(a.shape) == 1:
                a = a.reshape(-1, 1, 1)
            A = np.ascontiguousarray(a, dtype=np.float64)
            self.thisptr = new cm.MatrixCPU(&A[0,0,0], A.shape[2], A.shape[1], A.shape[0])
            self.A = A # make sure numpy array does not get GCed

    def __dealloc__(self):
        del self.thisptr

    cdef cm.MatrixView3DCPU get_standard_view(self):
        return self.thisptr.standard_view_3d

    def __len__(self):
        return self.thisptr.n_slices

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.thisptr[0][item]
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or len(self)
            return BufferView(self, start, stop)

    def print_me(self):
        self.thisptr.print_me()


cdef class BufferView:
    cdef cm.MatrixView2DCPU flatten2D(self):
        return self.view.flatten()

    def __cinit__(self, a=None, batches=1, features=1):
        if a is None:
            self.view = cm.MatrixView3DCPU()
        elif isinstance(a, int):
            self.B = Buffer(a * batches * features)
            self.view = cm.MatrixView3DCPU(features, batches, a)
            self.view.set_data(&self.B.thisptr[0][0])
        else: # a is np array
            self.B = Buffer(a)
            self.view = self.B.get_standard_view()

    def __len__(self):
        return self.view.size

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.view[item]
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or self.get_time_size()

            b = BufferView()
            b.view = self.view.slice(start, stop)
            b.B = self.B
            return b

    # from here: https://gist.github.com/1249305
    def as_array(self):
        cdef np.npy_intp shape[3]
        cdef np.ndarray ndarray
        shape[0] = <np.npy_intp> self.get_time_size()
        shape[1] = <np.npy_intp> self.get_batch_size()
        shape[2] = <np.npy_intp> self.get_feature_size()
        # Create a 3D array
        ndarray = np.PyArray_SimpleNewFromData(3, shape, np.NPY_FLOAT64, self.view.data)
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
        b = BufferView()
        b.view = cm.MatrixView3DCPU(feature_size, batch_size, time_size)
        b.view.set_data(&self.view[0])
        b.B = self.B
        return b


    def print_me(self):
        self.view.print_me()



def dot(BufferView a not None, BufferView b not None, BufferView out not None):
    cm.dot(a.flatten2D(), b.flatten2D(), out.flatten2D())

#def add(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
#    cm.add(a.flatten2D(), b.flatten2D(), out.flatten2D())

def add_into_b(BufferView a not None, BufferView b not None):
    cm.add_into_b(a.flatten2D(), b.flatten2D())

def add_scalar(BufferView a not None, double b):
    cm.add_scalar(a.flatten2D(), b)

def mult(BufferView a not None, BufferView b not None, BufferView out not None):
    cm.mult(a.flatten2D(), b.flatten2D(), out.flatten2D())

def mult_add(BufferView a not None, BufferView b not None, BufferView out not None):
    cm.mult_add(a.flatten2D(), b.flatten2D(), out.flatten2D())

def dot(BufferView a not None, BufferView b not None, BufferView out not None):
    cm.dot(a.flatten2D(), b.flatten2D(), out.flatten2D())

def dot_add(BufferView a not None, BufferView b not None, BufferView out not None):
    cm.dot_add(a.flatten2D(), b.flatten2D(), out.flatten2D())

def apply_sigmoid(BufferView a not None, BufferView out not None):
    cm.add_into_b(a.flatten2D(), out.flatten2D())

def apply_tanh(BufferView a not None, BufferView out not None):
    cm.add_into_b(a.flatten2D(), out.flatten2D())

def apply_tanhx2(BufferView a not None, BufferView out not None):
    cm.add_into_b(a.flatten2D(), out.flatten2D())

def equals(BufferView a not None, BufferView out not None):
    cm.add_into_b(a.flatten2D(), out.flatten2D())
