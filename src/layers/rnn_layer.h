/**
 * \file rnn_layer.h
 * \brief Declares the whole exception hierarchy.
 *
 * \details
 */


#ifndef __RNN_LAYER_H__
#define __RNN_LAYER_H__

#include "matrix/matrix_cpu.h"
#include <iostream>

struct RnnWeights {
  size_t n_inputs, n_cells;

  ///Variables defining sizes
  MatrixView2DCPU HX, HH;  //!< inputs X, H, S to input gate I 

  MatrixView2DCPU H_bias;   //!< bias to input gate, forget gate, state Z, output gate
  //MatrixCPU weights; 

  RnnWeights(size_t n_inputs, size_t n_cells);

  size_t buffer_size();
};

struct RnnBuffers {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, time;

  //Views on all activations
  MatrixView3DCPU Ha, Hb; //!< Hidden unit activation and output

  RnnBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
  
  size_t buffer_size();
};

struct RnnDeltas {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, time;

  //Views on all activations
  MatrixView3DCPU Ha, Hb; //Hidden unit activation and output

  MatrixView3DCPU temp_hidden, temp_hidden2; //temp values, neccessary? 

  RnnDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
  size_t buffer_size();
};

void rnn_forward(RnnWeights &w, RnnBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y);
void rnn_backward(RnnWeights &w, RnnBuffers &b, RnnDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas);

#endif
