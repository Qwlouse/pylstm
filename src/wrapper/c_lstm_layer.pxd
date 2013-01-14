from c_matrix cimport MatrixView2DCPU, MatrixView3DCPU

cdef extern from "lstm_layer.h":
    cdef cppclass LstmWeights:
        LstmWeights(int, int)
        int buffer_size()
        void allocate(MatrixView2DCPU)

    cdef cppclass LstmBuffers:
        LstmBuffers(int, int, int, int)
        int buffer_size()
        void allocate(MatrixView2DCPU)

    cdef cppclass LstmDeltas:
        LstmDeltas(int, int, int, int)
        int buffer_size()
        void allocate(MatrixView2DCPU)

    void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y)
