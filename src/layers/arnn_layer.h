#pragma once

#include <iostream>

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"
#include "matrix/matrix_container.h"

class ArnnLayer {
public:
    const ActivationFunction* f;
    ArnnLayer();
    explicit ArnnLayer(const ActivationFunction* f);
    ~ArnnLayer();

    class Parameters : public ::MatrixContainer {
    public:
        Matrix HX;
        Matrix HR;
        Matrix Timing;
        Matrix HR_tmp;
        Matrix H_bias;

        Parameters(size_t n_inputs, size_t n_cells);
    };

    class FwdState : public ::MatrixContainer{
    public:
        Matrix Ha; //!< total input activations for all neurons in layer

        FwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
    };

    class BwdState: public ::MatrixContainer {
    public:
        Matrix Ha; //!< activations for all neurons in layer
        Matrix Hb;
        Matrix tmp;

        BwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
    };

    void forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y);
    void backward(Parameters &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
    void gradient(Parameters &w, Parameters &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix &out_deltas);
    void Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
    void dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};
