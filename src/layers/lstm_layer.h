#pragma once

#include "matrix/matrix.h"
#include "layer.hpp"
#include <iostream>


class LstmLayer {
public:
	struct Weights : public ViewContainer {
		size_t n_inputs, n_cells;

		///Variables defining sizes
		Matrix IX, IH, IS;  //!< inputs X, H, S to input gate I
		Matrix FX, FH, FS;  //!< inputs X, H, S to forget gate F
		Matrix ZX, ZH;      //!< inputs X, H, to state cell
		Matrix OX, OH, OS;  //!< inputs X, H, S to output gate O

		Matrix I_bias, F_bias, Z_bias, O_bias;   //!< bias to input gate, forget gate, state Z, output gate

		Weights(size_t n_inputs, size_t n_cells);
	};

	struct FwdState : public ViewContainer {
	  ///Variables defining sizes
	  size_t n_inputs, n_cells;
	  size_t n_batches, time;

	  //Views on all activations
	  Matrix Ia, Ib; //!< Input gate activation
	  Matrix Fa, Fb; //!< forget gate activation
	  Matrix Oa, Ob; //!< output gate activation

	  Matrix Za, Zb; //!< Za =Net Activation, Zb=f(Za)
	  Matrix S;      //!< Sa =Cell State activations
	  Matrix f_S;      //!< Sa =Cell State activations
	  Matrix Hb;     //!< output of LSTM block
	  Matrix tmp1;     //!< tmp varin  LSTM block

	  FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	};

	struct BwdState : public ViewContainer {
	  ///Variables defining sizes
	  size_t n_inputs, n_cells;
	  size_t n_batches, time;

	  //Views on all activations
	  Matrix Ia, Ib; //Input gate activation
	  Matrix Fa, Fb; //forget gate activation
	  Matrix Oa, Ob; //output gate activation

	  Matrix Za, Zb; //Net Activation
	  Matrix S; //Cell activations
	  Matrix f_S; //cell state activations
	  Matrix Hb;     //!< output of LSTM block
	  Matrix tmp1;

	  BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
	};

	void forward(Weights &w, FwdState &b, Matrix &x, Matrix &y);
	void backward(Weights &w, FwdState &b, BwdState &d, Matrix &y, Matrix &in_deltas, Matrix &out_deltas);
	void gradient(Weights &w, Weights &grad, FwdState &b, BwdState &d, Matrix &y, Matrix& x, Matrix& out_deltas);
	void Rpass(Weights &w, Weights &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix &Ry);
	//void Rbackward(Weights &w, FwdState &b, BwdState &d, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu);
};

/*

void lstm_Rpass(LstmWeights &w, LstmWeights &v,  LstmBuffers &b, LstmBuffers &Rb, MatrixView3DCPU &x, MatrixView3DCPU &y, MatrixView3DCPU &Ry);
void lstm_Rbackward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas, LstmBuffers &Rb, double lambda, double mu);

*/

