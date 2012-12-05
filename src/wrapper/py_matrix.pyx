cimport c_matrix

cdef class MatrixCPU:
    def __cinit__(self, int n_rows, int n_columns, int n_slices):
        self.thisptr = new CMatrixCPU(n_rows, n_columns, n_slices)

    def __dealloc__(self):
        del self.thisptr
