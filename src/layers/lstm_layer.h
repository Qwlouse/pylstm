#ifndef __LSTM_LAYER_H__
#define __LSTM_LAYER_H__

#include "matrix/matrix_cpu.h"
#include <iostream>

struct LstmWeights {
  //matrix_type hidden_bias, output_bias;
  size_t n_input, n_cells, n_output;

  ///Variables defining sizes
  MatrixView2DCPU IX, IH, IS;
  MatrixView2DCPU FX, FH, FS;
  MatrixView2DCPU ZX, ZH;
  MatrixView2DCPU OX, OH, OS;

  MatrixView2DCPU I_bias, F_bias, Z_bias, O_bias;
  MatrixCPU weights;

  LstmWeights();
};

struct LstmBuffers {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, time;

  //Views on all activations
  MatrixView3DCPU Ia, Ib; //Input gate activation
  MatrixView3DCPU Fa, Fb; //forget gate activation
  MatrixView3DCPU Oa, Ob; //output gate activation

  MatrixView3DCPU Za, Zb; //Net Activation
  MatrixView3DCPU Sa, Sb; //Cell activations
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
  MatrixView3DCPU Sa, Sb; //Cell activations
};

void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y);

#endif
