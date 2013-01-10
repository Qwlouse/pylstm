from c_matrix cimport MatrixView2DCPU, MatrixView3DCPU

cdef extern from "lstm_layer.h":
    cdef cppclass LstmWeights:
        LstmWeights(int, int)
        buffer_size()
        allocate(MatrixView2DCPU&)

    cdef cppclass LstmBuffers:
        LstmBuffers(int, int, int, int)
        buffer_size()
        allocate(MatrixView2DCPU&)

    cdef cppclass LstmDeltas:
        LstmBuffers(int, int, int, int)
        buffer_size()
        allocate(MatrixView2DCPU&)

    void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y)
