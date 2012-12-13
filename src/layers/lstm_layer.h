/**
 * \file lstm_layer.h
 * \brief Declares the whole exception hierarchy.
 *
 * \details
 */


#ifndef __LSTM_LAYER_H__
#define __LSTM_LAYER_H__

#include "matrix/matrix_cpu.h"
#include <iostream>

struct LstmWeights {
  //matrix_type hidden_bias, output_bias;
  size_t n_input, n_cells, n_output;

  ///Variables defining sizes
  MatrixView2DCPU IX, IH, IS;  //!< inputs X, H, S to input gate I 
  MatrixView2DCPU FX, FH, FS;  //!< inputs X, H, S to forget gate F
  MatrixView2DCPU ZX, ZH;      //!< inputs X, H, to state cell 
  MatrixView2DCPU OX, OH, OS;  //!< inputs X, H, S to output gate O

  MatrixView2DCPU I_bias, F_bias, Z_bias, O_bias;   //!< bias to input gate, forget gate, state Z, output gate
  MatrixCPU weights; 

  LstmWeights();

  size_t size()
};

struct LstmBuffers {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, time;

  //Views on all activations
  MatrixView3DCPU Ia, Ib; //!< Input gate activation
  MatrixView3DCPU Fa, Fb; //!< forget gate activation
  MatrixView3DCPU Oa, Ob; //!< output gate activation

  MatrixView3DCPU Za, Zb; //!< Za =Net Activation, Zb=f(Za)
  MatrixView3DCPU S;      //!< Sa =Cell State activations
  MatrixView3DCPU f_S;      //!< Sa =Cell State activations
  MatrixView3DCPU Hb;     //!< output of LSTM block
};

struct LstmDeltas {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, batch_time;

  //Views on all activations
  MatrixView3DCPU Ia, Ib; //Input gate activation
  MatrixView3DCPU Fa, Fb; //forget gate activation
  MatrixView3DCPU Oa, Ob; //output gate activation

  MatrixView3DCPU Za, Zb; //Net Activation
  MatrixView3DCPU S; //Cell activations
  MatrixView3DCPU f_S; //cell state activations
  MatrixView3DCPU Hb;     //!< output of LSTM block


  MatrixView3DCPU temp_hidden, temp_hidden2; 
};

void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y);
void lstm_backward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas);

#endif
