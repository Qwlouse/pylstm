/**
 * \file fwd_layer.cpp
 * \brief Implementation of the fwd_layer.
 */


#include "fwd_layer.h"
#include "matrix/matrix_operation_cpu.h"

FwdWeights::FwdWeights(size_t n_inputs_, size_t n_cells_) :
  n_inputs(n_inputs_), 
  n_cells(n_cells_), 

  HX(n_cells, n_inputs), 
  H_bias(n_cells, 1); 
{
}

size_t FwdWeights::buffer_size() {
  return HX.size + H_bias.size;  //!< inputs X, H, to cells H + bias to H

}

FwdBuffers::FwdBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ha(n_cells, n_batches, time), Hb(n_cells, n_batches, time) //!< Input gate activation

{}

size_t FwdBuffers::buffer_size() {
 //Views on all activations
  return Ha.size + Hb.size; //!< Hidden unit activation
    
}

FwdDeltas::FwdDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  ///Variables defining sizes
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ha(n_cells, n_batches, time), Hb(n_cells, n_batches, time) //Input gate activation
 
  temp_hidden(n_cells, n_batches, time), temp_hidden2(n_cells, n_batches, time)
{}

size_t FwdDeltas::buffer_size() {
  return Ha.size + Hb.size + //Hidden unit activation
    temp_hidden.size + temp_hidden2.size; //temp vars
}

void fwd_forward(FwdWeights &w, FwdBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y) {
  mult(w.HX, x.flatten(), b.Ha.flatten());

  add_into_b(w.H_bias, b.Ha);
  apply_sigmoid(b.Ha, b.Hb);

  //for (size_t t(0); t < b.time; ++t) {
  //  add_into_b(w.H_bias, b.Ha.slice(t));
  // apply_sigmoid(b.Ha.slice(t), b.Hb.slice(t));     
  //}  
}

void fwd_backward(FwdWeights &w, FwdBuffers &b, FwdDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas) {

  size_t end_time(b.time - 1);


  mult(w.HH.T(), d.Hb.slice, d.Hb);

  //calculate t+1 values except for end_time+1 
  //for(int t(end_time); t >= 0; --t){
  //  if (t<end_time) {
  //    
  //    mult(w.HH.T(), d.Hb.slice(t+1), d.Hb.slice(t));
  //  }

    // \f$\frac{dE}{da_H} = \frac{dE}{db_H} * f'(a_H)\f$
    //sigmoid_deriv(d.Hb.slice(t), b.Hb.slice(t), d.temp_hidden, d.temp_hidden2, d.Ha.slice(t));    
   
  }
}

void fwd_grad(FwdWeights &w, FwdWeights &grad, FwdBuffers &b, FwdDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU input_batches)  {

  mult(d.Ha, input_batches.T(), grad.HX); 
  squash(d.Ha, grad.H_bias); 
  
}
