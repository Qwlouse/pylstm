cdef extern from "matrix_cpu.h":
    cdef cppclass MatrixCPU:
        MatrixCPU(int, int, int)


