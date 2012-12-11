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
        A = np.ascontiguousarray(array, dtype=np.float64)

        cdef np.npy_intp rows = A.shape[0]
        cdef np.npy_intp cols = A.shape[1]
        cdef np.npy_intp slices = A.shape[2]

        self.thisptr = new CMatrixCPU(&A[0,0,0], rows, cols, slices)
        self.A = A # make sure numpy array does not get GCed

    def __dealloc__(self):
        del self.thisptr

    def print_me(self):
        self.thisptr.print_me()