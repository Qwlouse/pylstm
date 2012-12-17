/**
 * \file rnn_layer.cpp
 * \brief Implementation of the rnn_layer.
 */


#include "rnn_layer.h"
#include "matrix/matrix_operation_cpu.h"

RnnWeights::RnnWeights(size_t n_inputs_, size_t n_cells_) :
  n_inputs(n_inputs_), 
  n_cells(n_cells_), 

  HX(n_cells, n_inputs), HH(n_cells, n_cells); 
 
H_bias(n_cells, 1); 
{
}

size_t RnnWeights::buffer_size() {
  return HX.size + HH.size + H_bias.size;  //!< inputs X, H, to cells H + bias to H

}

RnnBuffers::RnnBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ha(n_cells, n_batches, time), Hb(n_cells, n_batches, time) //!< Input gate activation

{}

size_t RnnBuffers::buffer_size() {
 //Views on all activations
  return Ha.size + Hb.size; //!< Hidden unit activation
    
}

RnnDeltas::RnnDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  ///Variables defining sizes
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ha(n_cells, n_batches, time), Hb(n_cells, n_batches, time) //Input gate activation
 
  temp_hidden(n_cells, n_batches, time), temp_hidden2(n_cells, n_batches, time)
{}

size_t RnnDeltas::buffer_size() {
  return Ha.size + Hb.size + //Hidden unit activation
    temp_hidden.size + temp_hidden2.size; //temp vars
}

void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y) {
  mult(w.HX, x.flatten(), b.Ha.flatten());

  for (size_t t(0); t < b.time; ++t) {
    //IF NEXT                                                                                                                                                                                                      
    if (t) {
      mult(w.HH, y.slice(t - 1), b.Ha.slice(t));  
    }

    add_into_b(w.H_bias, b.Ha.slice(t));

    apply_sigmoid(b.Ha.slice(t), b.Hb.slice(t));
      
  }  
}

void rnn_backward(RnnWeights &w, RnnBuffers &b, RnnDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas) {

  //clear_temp();
  //size_t end_time(b.batch_time - 1);
  size_t end_time(b.time - 1);

  //calculate t+1 values except for end_time+1 
  for(int t(end_time); t >= 0; --t){
    if (t<end_time) {
      
      mult(w.HH.T(), d.Hb.slice(t+1), d.Hb.slice(t));
    }

    // \f$\frac{dE}{da_H} = \frac{dE}{db_H} * f'(a_H)\f$
    //sigmoid_deriv(d.Hb.slice(t), b.Hb.slice(t), d.temp_hidden, d.temp_hidden2, d.Ha.slice(t));    
   
  }
}




