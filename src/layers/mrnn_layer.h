#pragma once

#include <iostream>

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"
#include "matrix/matrix_container.h"



class MrnnLayer {
public:
    const ActivationFunction* f;

    MrnnLayer();
    explicit MrnnLayer(const ActivationFunction* f);
    ~MrnnLayer();

    class Parameters : public ::MatrixContainer {
    public:
        Matrix HX;  // input X to hidden units U
        Matrix FX;  // input X to factor units F
        Matrix FH;  // hidden units H to factor units F
        Matrix HF;  // factor units F to hidden units H

        Matrix H_bias;

        Parameters(size_t n_inputs, size_t n_cells);
    };

    class FwdState : public ::MatrixContainer{
    public:
        Matrix Ha; // input activations for hidden units
        Matrix F1; // activations for factor units from inputs
        Matrix F2; // activations for factor units from recurrency
        Matrix Fa; // total input activations for factor units

        FwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
    };

    class BwdState: public ::MatrixContainer {
    public:
        Matrix Ha; // input activations for hidden units
        Matrix Hb; //
        Matrix F1; // activations for factor units from inputs
        Matrix F2; // activations for factor units from recurrency
        Matrix Fa; // total input activations for factor units

        BwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
    };

    void forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y);
    void backward(Parameters &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
    void gradient(Parameters &w, Parameters &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix &out_deltas);
    void Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
    void dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};
