/**
 * \file lstm_layer.cpp
 * \brief Implementation of the lstm_layer.
 */

#include <vector>

#include "lstm_layer.h"
#include "matrix/matrix_operation_cpu.h"

using namespace std;

LstmWeights::LstmWeights(size_t n_inputs_, size_t n_cells_) :
  n_inputs(n_inputs_), 
  n_cells(n_cells_), 

  IX(n_cells, n_inputs), IH(n_cells, n_cells), IS(n_cells, n_cells),
  FX(n_cells, n_inputs), FH(n_cells, n_cells), FS(n_cells, n_cells),
  ZX(n_cells, n_inputs), ZH(n_cells, n_cells),
  OX(n_cells, n_inputs), OH(n_cells, n_cells), OS(n_cells, n_cells),

  I_bias(n_cells, 1), F_bias(n_cells, 1), Z_bias(n_cells, 1), O_bias(n_cells, 1)
{
}

size_t LstmWeights::buffer_size() {
  return IX.size + IH.size + IS.size +  //!< inputs X, H, S to input gate I 
  FX.size + FH.size + FS.size +  //!< inputs X, H, S to forget gate F
  ZX.size + ZH.size +      //!< inputs X, H, to state cell 
  OX.size + OH.size + OS.size +  //!< inputs X, H, S to output gate O

  I_bias.size + F_bias.size + Z_bias.size + O_bias.size;   //!< bias to input gate, forget gate, state Z, output gate
}

void LstmWeights::allocate(MatrixView2DCPU buffer_view) {
  vector<MatrixView2DCPU*> views;

  views.push_back(&IX);
  views.push_back(&IH);
  views.push_back(&IS);
  views.push_back(&FX);
  views.push_back(&FH);
  views.push_back(&FS);
  views.push_back(&ZX);
  views.push_back(&ZH);
  views.push_back(&OX);
  views.push_back(&OH);
  views.push_back(&OS);
  
  views.push_back(&I_bias);
  views.push_back(&F_bias);
  views.push_back(&Z_bias);
  views.push_back(&O_bias);
  
  lay_out(buffer_view, views);
}

LstmBuffers::LstmBuffers(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ia(n_cells, n_batches, time), Ib(n_cells, n_batches, time), //!< Input gate activation
  Fa(n_cells, n_batches, time), Fb(n_cells, n_batches, time), //!< forget gate activation
  Oa(n_cells, n_batches, time), Ob(n_cells, n_batches, time), //!< output gate activation

  Za(n_cells, n_batches, time), Zb(n_cells, n_batches, time), //!< Za =Net Activation, Zb=f(Za)
  S(n_cells, n_batches, time),      //!< Sa =Cell State activations
  f_S(n_cells, n_batches, time),      //!< Sa =Cell State activations
  Hb(n_cells, n_batches, time)     //!< output of LSTM block
{}

size_t LstmBuffers::buffer_size() {
 //Views on all activations
  return Ia.size + Ib.size + //!< Input gate activation
    Fa.size + Fb.size + //!< forget gate activation
    Oa.size + Ob.size + //!< output gate activation
    
    Za.size + Zb.size + //!< Za =Net Activation, Zb=f(Za)
    S.size +      //!< Sa =Cell State activations
    f_S.size +      //!< Sa =Cell State activations
    Hb.size;     //!< output of LSTM block
}

void LstmBuffers::allocate(MatrixView2DCPU buffer_view) {
  vector<MatrixView3DCPU*> views;
  
  views.push_back(&Ia);
  views.push_back(&Ib); //Input gate activation
  views.push_back(&Fa);
  views.push_back(&Fb); //forget gate activation
  views.push_back(&Oa);
  views.push_back(&Ob); //output gate activation

  views.push_back(&Za);
  views.push_back(&Zb); //Net Activation
  views.push_back(&S); //Cell activations
  views.push_back(&f_S); //cell state activations
  views.push_back(&Hb);     //!< output of LSTM block

  lay_out(buffer_view, views);
}

LstmDeltas::LstmDeltas(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
  ///Variables defining sizes
  n_inputs(n_inputs_), n_cells(n_cells_),
  n_batches(n_batches_), time(time_),

  //Views on all activations
  Ia(n_cells, n_batches, time), Ib(n_cells, n_batches, time), //Input gate activation
  Fa(n_cells, n_batches, time), Fb(n_cells, n_batches, time), //forget gate activation
  Oa(n_cells, n_batches, time), Ob(n_cells, n_batches, time), //output gate activation

  Za(n_cells, n_batches, time), Zb(n_cells, n_batches, time), //Net Activation
  S(n_cells, n_batches, time), //Cell activations
  f_S(n_cells, n_batches, time), //cell state activations
  Hb(n_cells, n_batches, time),     //!< output of LSTM block

  temp_hidden(n_cells, n_batches, time), temp_hidden2(n_cells, n_batches, time)
{}

size_t LstmDeltas::buffer_size() {
  return Ia.size + Ib.size + //Input gate activation
    Fa.size + Fb.size + //forget gate activation
    Oa.size + Ob.size + //output gate activation
    
    Za.size + Zb.size + //Net Activation
    S.size + //Cell activations
    f_S.size + //cell state activations
    Hb.size +     //!< output of LSTM block
    temp_hidden.size + temp_hidden2.size;
}

void LstmDeltas::allocate(MatrixView2DCPU buffer_view) {
  vector<MatrixView3DCPU*> views;
  
  views.push_back(&Ia);
  views.push_back(&Ib); //Input gate activation
  views.push_back(&Fa);
  views.push_back(&Fb); //forget gate activation
  views.push_back(&Oa);
  views.push_back(&Ob); //output gate activation

  views.push_back(&Za);
  views.push_back(&Zb); //Net Activation
  views.push_back(&S); //Cell activations
  views.push_back(&f_S); //cell state activations
  views.push_back(&Hb);     //!< output of LSTM block

  views.push_back(&temp_hidden);
  views.push_back(&temp_hidden2);

  lay_out(buffer_view, views);
}

void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y) {
  
  mult(w.IX, x.flatten(), b.Ia.flatten());
  mult(w.FX, x.flatten(), b.Fa.flatten());
  mult(w.ZX, x.flatten(), b.Za.flatten());
  mult(w.OX, x.flatten(), b.Oa.flatten());

  for (size_t t(0); t < b.time; ++t) {
    //IF NEXT                                                                                 
    if (t) {

      mult(w.FH, y.slice(t - 1), b.Fa.slice(t));
      mult(w.IH, y.slice(t - 1), b.Ia.slice(t));
      mult(w.OH, y.slice(t - 1), b.Oa.slice(t));
      mult(w.ZH, y.slice(t - 1), b.Za.slice(t));
      
      mult_add(b.S.slice(t - 1), w.FS, b.Fa.slice(t));
      mult_add(b.S.slice(t - 1), w.IS, b.Ia.slice(t));
    }

    add_into_b(w.F_bias, b.Fa.slice(t));
    add_into_b(w.I_bias, b.Ia.slice(t));
    add_into_b(w.Z_bias, b.Za.slice(t));
    add_into_b(w.O_bias, b.Oa.slice(t));

    apply_sigmoid(b.Fa.slice(t), b.Fb.slice(t));
    apply_sigmoid(b.Ia.slice(t), b.Ib.slice(t));
    apply_sigmoid(b.Za.slice(t), b.Zb.slice(t));
    dot_add(b.Zb.slice(t), b.Ib.slice(t), b.S.slice(t));
    
    if (t) 
      dot_add(b.S.slice(t - 1), b.Fb.slice(t), b.S.slice(t));
    apply_tanhx2(b.S.slice(t), b.f_S.slice(t));

    b.f_S.slice(t).print_me();
    w.OS.print_me();
    b.Oa.slice(t).print_me();
    
    mult_add(b.S.slice(t), w.OS, b.Oa.slice(t));
    apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
    dot(b.f_S.slice(t), b.Ob.slice(t), y.slice(t));
  }
}

void lstm_backward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas) {

  //clear_temp();
  //size_t end_time(b.batch_time - 1);
  int end_time = static_cast<int>(b.time - 1);

  //calculate t+1 values except for end_time+1 
  for(int t(end_time); t >= 0; --t){
    if (t<end_time) {
      mult(w.IH.T(), d.Ia.slice(t+1), d.Hb.slice(t));
      mult(w.FH.T(), d.Fa.slice(t+1), d.Hb.slice(t));
      mult(w.ZH.T(), d.Za.slice(t+1), d.Hb.slice(t));
      mult(w.OH.T(), d.Oa.slice(t+1), d.Hb.slice(t));
      
      //! \f$\frac{dE}{dS} += \frac{dE}{dS^{t+1}} * b_F(t+1)\f$
      dot_add(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
      //! \f$\frac{dE}{dS} += \frac{dE}{da_I(t+1)} * W_{IS}\f$
      mult_add(d.Ia.slice(t+1), w.IS, d.S.slice(t));
      //! \f$\frac{dE}{dS} += \frac{dE}{da_F(t+1)} * W_{FS}\f$
      mult_add(d.Fa.slice(t+1), w.FS, d.S.slice(t));
    }

    //! \f$\frac{dE}{df_S} += \frac{dE}{dH} * b_O\f$  THIS IS WEIRD, IT GOES WITH NEXT LINE ??!?!
    dot_add(d.Hb.slice(t), b.Ob.slice(t), d.f_S.slice(t));
    
    //OUTPUT GATES DERIVS
    //! \f$\frac{dE}{db_O} = \frac{dE}{dH} * f(s) * f(a_O)\f$
    dot_add(d.Hb.slice(t), d.f_S.slice(t), d.Ob.slice(t));
    //! \f$\frac{dE}{da_O} = \frac{dE}{db_O} * f'(a_O)\f$
    //sigmoid_deriv(d.Ob.slice(t), b.Ob.slice(t), d.temp_hidden, d.temp_hidden2, d.Oa.slice(t)); //o = -o^2
      
    //! \f$\frac{dE}{dS} += \frac{dE}{df_S} * f'(s)\f$
    //tanh2_deriv(d.f_S.slice(t), b.S.slice(t), d.temp_hidden, d.S.slice(t));

    //! \f$\frac{dE}{dS} += \frac{dE}{da_O} * W_OS\f$
    mult_add(d.Oa.slice(t), w.OS, d.S.slice(t));
    
    //! CELL ACTIVATION DERIVS
    //! \f$\frac{dE}{db_Z} = \frac{dE}{dS} * b_I\f$
    dot_add(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
    //! \f$dE/da_Z = dE/db_Z * f'(a_Z)\f$
    //tanh2_deriv(d.Zb.slice(t), b.Zb.slice(t), d.temp_hidden, d.Za.slice(t));
    
    //! INPUT GATE DERIVS
    //! \f$\frac{dE}{db_I} = \frac{dE}{dS} * b_Z \f$
    dot_add(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));
    //! \f$\frac{dE}{da_I} = \frac{dE}{db_I} * f'(a_I) \f$
    //sigmoid_deriv(d.Ib.slice(t), b.Ib.slice(t), d.temp_hidden, d.temp_hidden2, d.Ia.slice(t));
      
    //! FORGET GATE DERIVS
    if (t)
      //! \f$\frac{dE}{db_F} += \frac{dE}{dS} * s(t-1)\f$
      dot_add(d.S.slice(t), b.S.slice(t - 1), d.Fb.slice(t));
    // \f$\frac{dE}{da_F} = \frac{dE}{db_F} * f'(a_F)\f$
    //sigmoid_deriv(d.Fb.slice(t), b.Fb.slice(t), d.temp_hidden, d.temp_hidden2, d.Fa.slice(t));    
   
  }
}


void lstm_grad(LstmWeights &w, LstmWeights &grad, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU input_batches, MatrixView3DCPU &in_deltas) {

  size_t n_updates(b.time);

  mult(d.output_deltas, d.Cb, delta_OT, 1.0 / n_updates);

  mult(d.Za, input_batches.flatten().T(), grad.ZX, 1.0 / n_updates);
  mult(d.Fa, input_batches.flatten().T(), grad.FX, 1.0 / n_updates);
  mult(d.Ia, input_batches.flatten().T(), grad.IX, 1.0 / n_updates);
  mult(d.Oa, input_batches.flatten().T(), grad.OX, 1.0 / n_updates);
  
  
  multiply_normal_transpose_shifted(d.Za, b.Hb, grad.ZH, d_n_batches, 1.0 / n_updates);
  multiply_normal_transpose_shifted(d.Fa, b.Hb, grad.FH, d_n_batches, 1.0 / n_updates);
  multiply_normal_transpose_shifted(d.Ia, b.Hb, grad.IH, d_n_batches, 1.0 / n_updates);
  multiply_normal_transpose_shifted(d.Oa, b.Hb, grad.OH, d_n_batches, 1.0 / n_updates);
  
  multiply_vector_shifted(d.Fa, b.S, grad.FS, d_n_batches);
  multiply_vector_shifted(d.Ia, b.S, grad.IS, d_n_batches);
  multiply_vector(d.Oa, b.S, grad.OS);
  
  mult(d.Fa.subslice(1, b.time).flatten(), b.S.subslice(0, b.time - 1).flatten().T(), grad.FS)
  multiply_vector_shifted(d.Fa, b.S, blah.FS, d_n_batches);
  multiply_vector_shifted(d.Ia, b.S, blah.IS, d_n_batches);
  multiply_vector(d.Oa, b.S, grad.OS);
  

  squash(grad.FS, delta_cF, 1.0 / n_updates);
  squash(grad.IS, delta_cI, 1.0 / n_updates);
  squash(grad.OS, delta_cO, 1.0 / n_updates);

  squash(d.Ia_deltas, delta_I_bias, 1.0 / n_updates);
  squash(d.Fa_deltas, delta_F_bias, 1.0 / n_updates);
  squash(d.Ca_deltas, delta_C_bias, 1.0 / n_updates);
  squash(d.Oa_deltas, delta_O_bias, 1.0 / n_updates);
  squash(d.output_deltas, delta_T_bias, 1.0 / n_updates);
  
  
}

/*
std::vector<element_type> calculate_gradient(LSTM_Weights<matrix_type> &lstm_weights, matrix3d_type &input_batches) {

    size_t n_updates(d_n_batches);
    bool do_copy(false);
    matrix_type delta_iI(lstm_weights.d_iI, do_copy), delta_hI(lstm_weights.d_hI, do_copy), delta_cI(lstm_weights.d_cI, do_copy);
    matrix_type delta_iF(lstm_weights.d_iF, do_copy), delta_hF(lstm_weights.d_hF, do_copy), delta_cF(lstm_weights.d_cF, do_copy);
    matrix_type delta_iC(lstm_weights.d_iC, do_copy), delta_hC(lstm_weights.d_hC, do_copy);
    matrix_type delta_iO(lstm_weights.d_iO, do_copy), delta_hO(lstm_weights.d_hO, do_copy), delta_cO(lstm_weights.d_cO, do_copy);
    matrix_type delta_OT(lstm_weights.d_OT, do_copy);

    matrix_type delta_I_bias(lstm_weights.d_I_bias, do_copy);
    matrix_type delta_F_bias(lstm_weights.d_F_bias, do_copy);
    matrix_type delta_C_bias(lstm_weights.d_C_bias, do_copy);
    matrix_type delta_O_bias(lstm_weights.d_O_bias, do_copy);
    matrix_type delta_T_bias(lstm_weights.d_T_bias, do_copy);

    multiply_normal_transpose(d_output_deltas, d_Cb, delta_OT, 1.0 / n_updates);
    multiply_normal_transpose(d_Ca_deltas, input_batches, delta_iC, 1.0 / n_updates);
    multiply_normal_transpose(d_Fa_deltas, input_batches, delta_iF, 1.0 / n_updates);
    multiply_normal_transpose(d_Ia_deltas, input_batches, delta_iI, 1.0 / n_updates);
    multiply_normal_transpose(d_Oa_deltas, input_batches, delta_iO, 1.0 / n_updates);
    multiply_normal_transpose_shifted(d_Ca_deltas, d_Cb, delta_hC, d_n_batches, 1.0 / n_updates);

    multiply_normal_transpose_shifted(d_Fa_deltas, d_Cb, delta_hF, d_n_batches, 1.0 / n_updates);
    multiply_normal_transpose_shifted(d_Ia_deltas, d_Cb, delta_hI, d_n_batches, 1.0 / n_updates);
    multiply_normal_transpose_shifted(d_Oa_deltas, d_Cb, delta_hO, d_n_batches, 1.0 / n_updates);

    multiply_vector_shifted(d_Fa_deltas, d_S, d_Fa_peephole_delta, d_n_batches);
    multiply_vector_shifted(d_Ia_deltas, d_S, d_Ia_peephole_delta, d_n_batches);
    multiply_vector(d_Oa_deltas, d_S, d_Oa_peephole_delta);

    squash(d_Fa_peephole_delta, delta_cF, 1.0 / n_updates);
    squash(d_Ia_peephole_delta, delta_cI, 1.0 / n_updates);
    squash(d_Oa_peephole_delta, delta_cO, 1.0 / n_updates);

    //		multiply_vector_shifted(d_Fa_deltas, d_S, delta_cF, d_n_batches, 1.0 / n_updates);
    //		multiply_vector_shifted(d_Ia_deltas, d_S, delta_cI, d_n_batches, 1.0 / n_updates);
    //		multiply_vector(d_Oa_deltas, d_S, delta_cO, 1.0 / n_updates);

    squash(d_Ia_deltas, delta_I_bias, 1.0 / n_updates);
    squash(d_Fa_deltas, delta_F_bias, 1.0 / n_updates);
    squash(d_Ca_deltas, delta_C_bias, 1.0 / n_updates);
    squash(d_Oa_deltas, delta_O_bias, 1.0 / n_updates);
    squash(d_output_deltas, delta_T_bias, 1.0 / n_updates);

    //		matrix_type delta_ih(LSTM_weights.d_ih), delta_hh(LSTM_weights.d_hh), delta_ht(LSTM_weights.d_ht),
    //				delta_hidden_bias(LSTM_weights.d_hidden_bias), delta_output_bias(LSTM_weights.d_output_bias);
    //		delta_ih.clear();
    //		delta_hh.clear();
    //		delta_ht.clear();
    //		delta_hidden_bias.clear();
    //		delta_output_bias.clear();
    //
    //		multiply_normal_transpose(d_output_deltas, d_hidden_activations, delta_ht, 1.0 / n_updates);
    //		multiply_normal_transpose(d_hidden_deltas, input_batches, delta_ih, 1.0 / n_updates);
    //		multiply_normal_transpose_shifted(d_hidden_deltas, d_hidden_activations, delta_hh, d_n_batches, 1.0 / n_updates);
    //
    //		squash(d_hidden_deltas, delta_hidden_bias, 1.0 / n_updates);
    //		squash(d_output_deltas, delta_output_bias, 1.0 / n_updates);
    //
    std::vector<element_type> gradient_values(lstm_weights.size());
    std::vector<element_type>::iterator gradient_it(gradient_values.begin());

    gradient_it = add_and_advance(delta_iI, gradient_it);
    gradient_it = add_and_advance(delta_hI, gradient_it);
    gradient_it = add_and_advance(delta_cI, gradient_it);
    gradient_it = add_and_advance(delta_iF, gradient_it);
    gradient_it = add_and_advance(delta_hF, gradient_it);
    gradient_it = add_and_advance(delta_cF, gradient_it);
    gradient_it = add_and_advance(delta_iC, gradient_it);
    gradient_it = add_and_advance(delta_hC, gradient_it);
    gradient_it = add_and_advance(delta_iO, gradient_it);
    gradient_it = add_and_advance(delta_hO, gradient_it);
    gradient_it = add_and_advance(delta_cO, gradient_it);
    gradient_it = add_and_advance(delta_OT, gradient_it);

    gradient_it = add_and_advance(delta_I_bias, gradient_it);
    gradient_it = add_and_advance(delta_F_bias, gradient_it);
    gradient_it = add_and_advance(delta_C_bias, gradient_it);
    gradient_it = add_and_advance(delta_O_bias, gradient_it);
    gradient_it = add_and_advance(delta_T_bias, gradient_it);
    
    //exit(1);

    //		copy(delta_ih, gradient_it);
    //		gradient_it += delta_ih.size();
    //		copy(delta_hh, gradient_it);
    //		gradient_it += delta_hh.size();
    //		copy(delta_ht, gradient_it);
    //		gradient_it += delta_ht.size();
    //		copy(delta_hidden_bias, gradient_it);
    //		gradient_it += delta_hidden_bias.size();
    //		copy(delta_output_bias, gradient_it);
    //
    return gradient_values;
  }

*/
