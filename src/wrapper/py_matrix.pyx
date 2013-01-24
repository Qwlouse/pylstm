import numpy as np
cimport numpy as np

cimport c_matrix as cm

# http://stackoverflow.com/questions/3046305
# http://article.gmane.org/gmane.comp.python.cython.user/5625

cdef class Buffer:
    cdef cm.MatrixCPU *thisptr      # hold a C++ instance which we're wrapping
    cdef np.ndarray A

    def __cinit__(self, a):
        cdef np.ndarray[np.double_t, ndim=1, mode='c'] A
        if isinstance(a, int):
            self.thisptr = new cm.MatrixCPU(1, 1, a)
        else:
            A = np.ascontiguousarray(a, dtype=np.float64).flatten()
            self.thisptr = new cm.MatrixCPU(&A[0], 1, 1, A.shape[0])
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
    cdef cm.MatrixView3DCPU view
    cdef Buffer B

    def __cinit__(self, a=None, batches=1, features=1):
        if a is None:
            self.view = cm.MatrixView3DCPU()
        else:
            self.B = Buffer(a)
            self.view = self.B.get_standard_view()

        if isinstance(a, int):
            self.view = cm.MatrixView3DCPU(features, batches, a)
            self.view.set_data(&self.B.thisptr[0][0])

    def __len__(self):
        return self.view.size

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.view[item]
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or len(self)

            b = BufferView()
            b.view = self.view.slice(start, stop)
            b.B = self.B
            return b

    def print_me(self):
        self.view.print_me()

    cdef cm.MatrixView2DCPU flatten2D(self):
        return self.view.flatten()



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
