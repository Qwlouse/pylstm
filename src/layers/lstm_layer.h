#pragma once

#include <iostream>

#include "matrix/matrix.h"
#include "matrix/matrix_operation.h"
#include "matrix/matrix_container.h"


class LstmLayer {
public:
    const ActivationFunction* f;

    // this value is used for clipping the deltas in the backprop before the
    // activation function to the range [-delta_range, delta_range].
    float delta_range;

	LstmLayer();
	explicit LstmLayer(const ActivationFunction* f);

	class Parameters : public MatrixContainer {
	public:
		///Variables defining sizes
		Matrix IX, IH, IS;  //!< inputs X, H, S to input gate I
		Matrix FX, FH, FS;  //!< inputs X, H, S to forget gate F
		Matrix ZX, ZH;      //!< inputs X, H, to state cell
		Matrix OX, OH, OS;  //!< inputs X, H, S to output gate O

		Matrix I_bias, F_bias, Z_bias, O_bias;   //!< bias to input gate, forget gate, state Z, output gate

		Parameters(size_t n_inputs, size_t n_cells);
	};

	class FwdState : public MatrixContainer {
	public:
	  //Views on all activations
	  Matrix Ia, Ib; //!< Input gate activation
	  Matrix Fa, Fb; //!< forget gate activation
	  Matrix Oa, Ob; //!< output gate activation

	  Matrix Za, Zb; //!< Za =Net Activation, Zb=f(Za)
	  Matrix S;      //!< Sa =Cell State activations
	  Matrix f_S;      //!< Sa =Cell State activations
	  Matrix Hb;     //!< output of LSTM block
	  Matrix tmp1;     //!< tmp varin  LSTM block

	  FwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
	};

	class BwdState : public MatrixContainer {
	public:
	  //Views on all activations
	  Matrix Ia, Ib; //Input gate activation
	  Matrix Fa, Fb; //forget gate activation
	  Matrix Oa, Ob; //output gate activation

	  Matrix Za, Zb; //Net Activation
	  Matrix S; //Cell activations
	  Matrix f_S; //cell state activations
	  Matrix Hb;     //!< output of LSTM block
	  Matrix tmp1;

	  BwdState(size_t n_inputs, size_t n_cells, size_t n_batches, size_t time);
	};

	void forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y, bool training_pass);
	void backward(Parameters &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
	void gradient(Parameters &w, Parameters &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix& out_deltas);
	void Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry);
	void dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};
