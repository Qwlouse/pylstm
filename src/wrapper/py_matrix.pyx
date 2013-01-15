import numpy as np
cimport numpy as np

cimport c_matrix as cm

# http://stackoverflow.com/questions/3046305
# http://article.gmane.org/gmane.comp.python.cython.user/5625

# forward declaration
cdef class MatrixCPU

cdef class MatrixView:
    cdef cm.MatrixView3DCPU view
    cdef MatrixCPU mat
    def __cinit__(self, MatrixCPU mat, start=0, stop=-1):
        if (start, stop) == (0, -1):
            self.view = mat.thisptr.standard_view_3d
        else:
            self.view = mat.thisptr.standard_view_3d.slice(start, stop)
        self.mat = mat

    def slice(self, int start, int stop):
        return MatrixView(self.mat, start, stop)

    def get_feature_count(self):
        return self.mat.get_feature_count()

    def get_batch_count(self):
        return self.mat.get_batch_count()

    def get_slice_count(self):
        return self.view.n_slices

    cdef cm.MatrixView2DCPU flatten2D(self):
        return self.view.flatten()

    def assign(self, a):
        cdef np.ndarray[np.double_t, ndim=3, mode='c'] A
        A = np.ascontiguousarray(a, dtype=np.float64)
        assert A.shape[0] == self.view.n_slices, "nr_slices mismatch"
        assert A.shape[1] == self.view.n_columns, "nr_columns mismatch"
        assert A.shape[2] == self.view.n_rows, "nr_rows mismatch"
        self.mat = MatrixCPU(A)
        self.view = self.mat.thisptr.standard_view_3d

    def print_me(self):
        self.view.print_me()


cdef class MatrixCPU:
    cdef cm.MatrixCPU *thisptr      # hold a C++ instance which we're wrapping
    cdef np.ndarray A
    
    cdef cm.MatrixView2DCPU get_2d_view(self):
        return self.thisptr.standard_view_2d

    cdef cm.MatrixView3DCPU get_3d_view(self):
        return self.thisptr.standard_view_3d

    def __cinit__(self, a, batch_size=0, time_size=0):
        cdef np.npy_intp rows
        cdef np.npy_intp cols
        cdef np.npy_intp slices
        if batch_size > 0:
            assert time_size > 0
            assert int(a) > 0
            # initialize with a size only
            rows = time_size
            cols = batch_size
            slices = a
            self.thisptr = new cm.MatrixCPU(rows, cols, slices)
            return

        # declare a 2d NumPy array in C order
        cdef np.ndarray[np.double_t, ndim=3, mode='c'] A
        # unbox NumPy array into numpy 3d member array A
        # make sure we have a contiguous array in C order
        # this might produce a copy
        A = np.ascontiguousarray(a, dtype=np.float64)

        rows = A.shape[2]
        cols = A.shape[1]
        slices = A.shape[0]

        self.thisptr = new cm.MatrixCPU(&A[0,0,0], rows, cols, slices)
        self.A = A # make sure numpy array does not get GCed

    def __dealloc__(self):
        del self.thisptr

    def get_view(self):
        return MatrixView(self, 0, self.get_slice_count())

    def print_me(self):
        self.thisptr.print_me()

    def get_feature_count(self):
        return self.thisptr.n_rows

    def get_batch_count(self):
        return self.thisptr.n_columns

    def get_slice_count(self):
        return self.thisptr.n_slices

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
