import numpy as np
cimport numpy as np
cimport c_matrix

# http://stackoverflow.com/questions/3046305
# http://article.gmane.org/gmane.comp.python.cython.user/5625

cdef class MatrixCPU:
    def __cinit__(self, object array):
        # declare a 2d NumPy array in C order
        cdef np.ndarray[np.double_t, ndim=3, mode='c'] A
        # unbox NumPy array into numpy 3d member array A
        # make sure we have a contiguous array in C order
        # this might produce a temporary copy
        A = np.ascontiguousarray(np.swapaxes(array, 1, 2), dtype=np.float64)

        cdef np.npy_intp rows = A.shape[2]
        cdef np.npy_intp cols = A.shape[1]
        cdef np.npy_intp slices = A.shape[0]

        self.thisptr = new cm.MatrixCPU(&A[0,0,0], rows, cols, slices)
        self.A = A # make sure numpy array does not get GCed

    def __dealloc__(self):
        del self.thisptr

    def print_me(self):
        self.thisptr.print_me()

def dot(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
    cm.dot(a.get_2d_view(), b.get_2d_view(), out.get_2d_view())

#def add(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
#    cm.add(a.get_2d_view(), b.get_2d_view(), out.get_2d_view())

def add_into_b(MatrixCPU a not None, MatrixCPU b not None):
    cm.add_into_b(a.get_2d_view(), b.get_2d_view())

def add_scalar(MatrixCPU a not None, double b):
    cm.add_scalar(a.get_2d_view(), b)

def mult(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
    cm.mult(a.get_2d_view(), b.get_2d_view(), out.get_2d_view())

def mult_add(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
    cm.mult_add(a.get_2d_view(), b.get_2d_view(), out.get_2d_view())

def dot(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
    cm.dot(a.get_2d_view(), b.get_2d_view(), out.get_2d_view())

def dot_add(MatrixCPU a not None, MatrixCPU b not None, MatrixCPU out not None):
    cm.dot_add(a.get_2d_view(), b.get_2d_view(), out.get_2d_view())

def apply_sigmoid(MatrixCPU a not None, MatrixCPU out not None):
    cm.add_into_b(a.get_2d_view(), out.get_2d_view())

def apply_tanh(MatrixCPU a not None, MatrixCPU out not None):
    cm.add_into_b(a.get_2d_view(), out.get_2d_view())

def apply_tanhx2(MatrixCPU a not None, MatrixCPU out not None):
    cm.add_into_b(a.get_2d_view(), out.get_2d_view())

def equals(MatrixCPU a not None, MatrixCPU out not None):
    cm.add_into_b(a.get_2d_view(), out.get_2d_view())