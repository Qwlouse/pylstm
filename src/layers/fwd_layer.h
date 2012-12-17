/**
 * \file fwd_layer.h
 * \brief Declares the whole exception hierarchy.
 *
 * \details
 */


#ifndef __FWD_LAYER_H__
#define __FWD_LAYER_H__

#include "matrix/matrix_cpu.h"
#include <iostream>

struct FwdWeights {
  size_t n_inputs, n_cells;

  ///Variables defining sizes
  MatrixView2DCPU HX;  //!< inputs X, H, S to input gate I 

  MatrixView2DCPU H_bias;   //!< bias to input gate, forget gate, state Z, output gate
  //MatrixCPU weights; 

  FwdWeights(size_t n_inputs, size_t n_cells);

  size_t buffer_size();
};

struct FwdBuffers {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, time;

  //Views on all activations
  MatrixView3DCPU Ha, Hb; //!< Hidden unit activation and output

  FwdBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
  
  size_t buffer_size();
};

struct FwdDeltas {
  ///Variables defining sizes
  size_t n_inputs, n_outputs, n_cells;
  size_t n_batches, time;

  //Views on all activations
  MatrixView3DCPU Ha, Hb; //Hidden unit activation and output

  MatrixView3DCPU temp_hidden, temp_hidden2; //temp values, neccessary? 

  FwdDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches, size_t time_);
  size_t buffer_size();
};

void fwd_forward(FwdWeights &w, FwdBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y);
void fwd_backward(FwdWeights &w, FwdBuffers &b, FwdDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas);

#endif
