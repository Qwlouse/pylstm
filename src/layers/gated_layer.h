#pragma once

#include <iostream>

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"
#include "matrix/matrix_container.h"


class GatedLayer {
public:
    const ActivationFunction* f;

    // this value is used for clipping the deltas in the backprop before the
    // activation function to the range [-delta_range, delta_range].
    float delta_range;

	GatedLayer();
	explicit GatedLayer(const ActivationFunction* f);

    const ActivationFunction* input_act_func;

	class Parameters : public MatrixContainer {
	public:
		///Variables defining sizes
		Matrix IX;  //!< inputs X, H, S to input gate I
		Matrix ZX;      //!< inputs X, H, to state cell

		Matrix I_bias, Z_bias;   //!< bias to input gate, forget gate, state Z, output gate

		Parameters(size_t n_inputs, size_t n_cells);
	};

	class FwdState : public MatrixContainer {
	public:
      Matrix S_last;
	  //Views on all activations
	  Matrix Ia, Ib; //!< Input gate activation
	  Matrix Za, Zb; //!< Za =Net Activation, Zb=f(Za)
	  Matrix Fb;
	  Matrix tmp1;     //!< tmp varin  LSTM block

	  FwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
	};

	class BwdState : public MatrixContainer {
	public:
	  //Views on all activations
	  Matrix Ia, Ib; //Input gate activation
	  Matrix Za, Zb; //Net Activation

	  Matrix tmp1;

	  BwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
	};

	void forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y, bool training_pass);
	void backward(Parameters &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
	void gradient(Parameters &w, Parameters &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix& out_deltas);
	void Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
	void dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};
