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

  IX(n_cells, n_inputs), IH(n_cells, n_cells), IS(1, n_cells),
  FX(n_cells, n_inputs), FH(n_cells, n_cells), FS(1, n_cells),
  ZX(n_cells, n_inputs), ZH(n_cells, n_cells),
  OX(n_cells, n_inputs), OH(n_cells, n_cells), OS(1, n_cells),

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
  Hb(n_cells, n_batches, time),     //!< output of LSTM block
  tmp1(n_cells, n_batches, time) // for calculating derivs
  
{}

size_t LstmBuffers::buffer_size() {
 //Views on all activations
  return Ia.size + Ib.size + //!< Input gate activation
    Fa.size + Fb.size + //!< forget gate activation
    Oa.size + Ob.size + //!< output gate activation
    
    Za.size + Zb.size + //!< Za =Net Activation, Zb=f(Za)
    S.size +      //!< Sa =Cell State activations
    f_S.size +      //!< Sa =Cell State activations
    Hb.size +       //!< output of LSTM block 
    tmp1.size; 
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
  
  views.push_back(&tmp1);
  
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
  
  tmp1(n_cells, n_batches, time) // for calculating derivs

  //temp_hidden(n_cells, n_batches, time), temp_hidden2(n_cells, n_batches, time)
{}

size_t LstmDeltas::buffer_size() {
  return Ia.size + Ib.size + //Input gate activation
    Fa.size + Fb.size + //forget gate activation
    Oa.size + Ob.size + //output gate activation
    
    Za.size + Zb.size + //Net Activation
    S.size + //Cell activations
    f_S.size + //cell state activations
    Hb.size +     //!< output of LSTM block
    tmp1.size; //temp_hidden.size + temp_hidden2.size;
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

  views.push_back(&tmp1);
  //views.push_back(&temp_hidden);
  //views.push_back(&temp_hidden2);

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
  
      mult_add(w.FH, y.slice(t - 1), b.Fa.slice(t));
      mult_add(w.IH, y.slice(t - 1), b.Ia.slice(t));
      mult_add(w.OH, y.slice(t - 1), b.Oa.slice(t));
      mult_add(w.ZH, y.slice(t - 1), b.Za.slice(t));
  
      dot_add(b.S.slice(t - 1), w.FS, b.Fa.slice(t));
      dot_add(b.S.slice(t - 1), w.IS, b.Ia.slice(t));
    }

    add_vector_into(w.F_bias, b.Fa.slice(t));
    add_vector_into(w.I_bias, b.Ia.slice(t));
    add_vector_into(w.Z_bias, b.Za.slice(t));
    add_vector_into(w.O_bias, b.Oa.slice(t));

    apply_sigmoid(b.Fa.slice(t), b.Fb.slice(t));
    apply_sigmoid(b.Ia.slice(t), b.Ib.slice(t));
    apply_tanhx2(b.Za.slice(t), b.Zb.slice(t));
    //dot_add(b.Zb.slice(t), b.Ib.slice(t), b.S.slice(t));
    dot(b.Zb.slice(t), b.Ib.slice(t), b.S.slice(t));

    if (t) 
      dot_add(b.S.slice(t - 1), b.Fb.slice(t), b.S.slice(t));
    apply_tanhx2(b.S.slice(t), b.f_S.slice(t));

    dot_add(b.S.slice(t), w.OS, b.Oa.slice(t));

    //mult_add(b.S.slice(t), w.OS, b.Oa.slice(t));
    apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
    //copy(b.Oa.slice(t), b.Ob.slice(t));

    dot(b.f_S.slice(t), b.Ob.slice(t), y.slice(t));
    
   }
}

void lstm_backward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas) {

  //clear_temp();
  //size_t end_time(b.batch_time - 1);
  int end_time = static_cast<int>(b.time - 1);

  copy(out_deltas, d.Hb);
  
  //calculate t+1 values except for end_time+1 
  for(int t(end_time); t >= 0; --t){
    if (t<end_time) { 
    
      mult_add(w.IH.T(), d.Ia.slice(t+1), d.Hb.slice(t));
      mult_add(w.FH.T(), d.Fa.slice(t+1), d.Hb.slice(t));
      mult_add(w.ZH.T(), d.Za.slice(t+1), d.Hb.slice(t));
      mult_add(w.OH.T(), d.Oa.slice(t+1), d.Hb.slice(t));
  
      //! \f$\frac{dE}{dS} += \frac{dE}{dS^{t+1}} * b_F(t+1)\f$
      dot_add(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
      
      //! \f$\frac{dE}{dS} += \frac{dE}{da_I(t+1)} * W_{IS}\f$
      dot_add(d.Ia.slice(t+1), w.IS, d.S.slice(t));

      //! \f$\frac{dE}{dS} += \frac{dE}{da_F(t+1)} * W_{FS}\f$
      dot_add(d.Fa.slice(t+1), w.FS, d.S.slice(t)); 

    }

    //! \f$\frac{dE}{df_S} += \frac{dE}{dH} * b_O\f$  THIS IS WEIRD, IT GOES WITH NEXT LINE ??!?!
    dot(d.Hb.slice(t), b.Ob.slice(t), d.f_S.slice(t));
    
  
    //OUTPUT GATES DERIVS
    //! \f$\frac{dE}{db_O} = \frac{dE}{dH} * f(s) * f(a_O)\f$
    dot(d.Hb.slice(t), b.f_S.slice(t), d.Ob.slice(t));

    //! \f$\frac{dE}{da_O} = \frac{dE}{db_O} * f'(a_O)\f$
    apply_sigmoid_deriv(b.Ob.slice(t), d.tmp1.slice(t)); //s'(O_a) == s(O_b) * (1 - s(O_b)) 
    dot(d.Ob.slice(t), d.tmp1.slice(t), d.Oa.slice(t));


    //! \f$\frac{dE}{dS} += \frac{dE}{df_S} * f'(s)\f$
    apply_tanhx2_deriv(b.S.slice(t), d.tmp1.slice(t));
    dot_add(d.f_S.slice(t), d.tmp1.slice(t), d.S.slice(t));


    //! \f$\frac{dE}{dS} += \frac{dE}{da_O} * W_OS\f$
    dot_add(d.Oa.slice(t), w.OS, d.S.slice(t));
    
    //! CELL ACTIVATION DERIVS
    //! \f$\frac{dE}{db_Z} = \frac{dE}{dS} * b_I\f$
    dot(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
    //! \f$dE/da_Z = dE/db_Z * f'(a_Z)\f$
    apply_tanhx2_deriv(b.Za.slice(t), d.tmp1.slice(t));
    dot(d.Zb.slice(t), d.tmp1.slice(t), d.Za.slice(t));
    
    //! INPUT GATE DERIVS
    //! \f$\frac{dE}{db_I} = \frac{dE}{dS} * b_Z \f$
    dot(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));

    //! \f$\frac{dE}{da_I} = \frac{dE}{db_I} * f'(a_I) \f$
    //sigmoid_deriv(d.Ib.slice(t), b.Ib.slice(t), d.temp_hidden, d.temp_hidden2, d.Ia.slice(t));
    
    //apply_sigmoid_deriv(b.Ib.slice(t), d.Ia.slice(t));
    apply_sigmoid_deriv(b.Ib.slice(t), d.tmp1.slice(t));
    dot(d.Ib.slice(t), d.tmp1.slice(t), d.Ia.slice(t));
 
   //! FORGET GATE DERIVS
    if (t)
      //! \f$\frac{dE}{db_F} += \frac{dE}{dS} * s(t-1)\f$
      dot(d.S.slice(t), b.S.slice(t - 1), d.Fb.slice(t));
    
    // \f$\frac{dE}{da_F} = \frac{dE}{db_F} * f'(a_F)\f$
    apply_sigmoid_deriv(b.Fb.slice(t), d.tmp1.slice(t));    
    dot(d.Fb.slice(t), d.tmp1.slice(t), d.Fa.slice(t));

    //dE/dx 
    mult(w.IX.T(), d.Ia.slice(t), in_deltas.slice(t));
    mult_add(w.ZX.T(), d.Za.slice(t), in_deltas.slice(t));
    mult_add(w.FX.T(), d.Fa.slice(t), in_deltas.slice(t));    
  }
}

//void lstm_grad(LstmWeights &w, LstmWeights &grad, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU input_batches, MatrixView3DCPU &in_deltas) {
void lstm_grad(LstmWeights &w, LstmWeights &grad, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU input_batches)  {

  size_t n_time(b.time);

  //mult(d.output_deltas, d.Cb, delta_OT, 1.0 / n_time);

  //! \f$\frac{dE}{dW_ZX} += \frac{dE}{da_Z} * x(t)\f$
  //! \f$\frac{dE}{dW_FX} += \frac{dE}{da_F} * x(t)\f$
  //! \f$\frac{dE}{dW_IX} += \frac{dE}{da_I} * x(t)\f$
  //! \f$\frac{dE}{dW_OX} += \frac{dE}{da_O} * x(t)\f$
  mult(d.Za, input_batches.T(), grad.ZX); //  1.0 / 1.0); //(double) n_time);
  mult(d.Fa, input_batches.T(), grad.FX); // 1.0 / 1.0); //(double) n_time);
  mult(d.Ia, input_batches.T(), grad.IX); //1.0 / 1.0); //(double) n_time);
  mult(d.Oa, input_batches.T(), grad.OX); // 1.0 / 1.0); //(double) n_time);
  
  //! \f$\frac{dE}{dW_ZH} += \frac{dE}{da_Z} * h(t-1)\f$
  //! \f$\frac{dE}{dW_FH} += \frac{dE}{da_F} * h(t-1)\f$
  //! \f$\frac{dE}{dW_IH} += \frac{dE}{da_I} * h(t-1)\f$
  //! \f$\frac{dE}{dW_OH} += \frac{dE}{da_O} * h(t-1)\f$						       
  if (n_time > 1) {
    mult(d.Ia.subslice(1, n_time-1), y.subslice(0, n_time - 2).T(), grad.IH); //(double) n_time);
    mult(d.Za.subslice(1, n_time-1), y.subslice(0, n_time - 2).T(), grad.ZH); //(double) n_time);
    mult(d.Fa.subslice(1, n_time-1), y.subslice(0, n_time - 2).T(), grad.FH); //(double) n_time);
    mult(d.Oa.subslice(1, n_time-1), y.subslice(0, n_time - 2).T(), grad.OH); //(double) n_time);
  }

 
  //! \f$\frac{dE}{dW_FS} += \frac{dE}{da_F} * s(t-1)\f$
  //! \f$\frac{dE}{dW_IS} += \frac{dE}{da_I} * s(t-1)\f$
  if (n_time > 1) {
    dot_squash(d.Fa.subslice(1, n_time-1), b.S.subslice(0, n_time - 2), grad.FS);
    dot_squash(d.Ia.subslice(1, n_time-1), b.S.subslice(0, n_time - 2), grad.IS);
  }

  //! \f$\frac{dE}{dW_OS} += \frac{dE}{da_O} * s(t)\f$
  dot_squash(d.Oa, b.S, grad.OS);
  
  squash(d.Ia, grad.I_bias); //, 1.0 / (double) n_time);
  squash(d.Fa, grad.F_bias); //, 1.0 / (double) n_time);
  squash(d.Za, grad.Z_bias); //, 1.0 / (double) n_time);
  squash(d.Oa, grad.O_bias); //, 1.0 / (double)n_time); //, 1.0 / (double) n_time);

}


void lstm_Rpass(LstmWeights &w, LstmWeights &v,  LstmBuffers &b, LstmBuffers &Rb, MatrixView3DCPU &x, MatrixView3DCPU &y, MatrixView3DCPU &Ry) {


  mult(w.IX, x.flatten(), Rb.Ia.flatten());
  mult(w.FX, x.flatten(), Rb.Fa.flatten());
  mult(w.ZX, x.flatten(), Rb.Za.flatten());
  mult(w.OX, x.flatten(), Rb.Oa.flatten());

  for (size_t t(0); t < b.time; ++t) {
    
    //IF NEXT                                                                                 
    if (t) { 
  
      mult_add(w.FH, Ry.slice(t - 1), Rb.Fa.slice(t));
      mult_add(w.IH, Ry.slice(t - 1), Rb.Ia.slice(t));
      mult_add(w.OH, Ry.slice(t - 1), Rb.Oa.slice(t));
      mult_add(w.ZH, Ry.slice(t - 1), Rb.Za.slice(t));

      mult_add(v.FH, y.slice(t - 1), Rb.Fa.slice(t));
      mult_add(v.IH, y.slice(t - 1), Rb.Ia.slice(t));
      mult_add(v.OH, y.slice(t - 1), Rb.Oa.slice(t));
      mult_add(v.ZH, y.slice(t - 1), Rb.Za.slice(t));
  
      dot_add(Rb.S.slice(t - 1), w.FS, Rb.Fa.slice(t));
      dot_add(Rb.S.slice(t - 1), w.IS, Rb.Ia.slice(t));
      dot_add(b.S.slice(t - 1), v.FS, Rb.Fa.slice(t));
      dot_add(b.S.slice(t - 1), v.IS, Rb.Ia.slice(t));
    }

    /// CHECK THIS SHIT!!
    //add_vector(d_RFa.matrix_from_slice(t), v.d_F_bias);
    //add_vector(d_RIa.matrix_from_slice(t), v.d_I_bias);
    //add_vector(d_RCa.matrix_from_slice(t), v.d_C_bias);
    add_vector_into(w.F_bias, b.Fa.slice(t));
    add_vector_into(w.I_bias, b.Ia.slice(t));
    add_vector_into(w.Z_bias, b.Za.slice(t));
    add_vector_into(w.O_bias, b.Oa.slice(t));
    

    //double check form of simoid deriv, should it get b.Ia or b.Ib?
    apply_sigmoid_deriv(b.Ia.slice(t), Rb.tmp1.slice(t));
    //shout this next line be dot_add?
    mult_add(Rb.tmp1.slice(t), Rb.Ia.slice(t), Rb.Ib.slice(t));

    //double check form of simoid deriv, should it get b.Ia or b.Ib?
    apply_sigmoid_deriv(b.Fa.slice(t), Rb.tmp1.slice(t));
    //shout this next line be dot_add?
    mult_add(Rb.tmp1.slice(t), Rb.Fa.slice(t), Rb.Fb.slice(t));
    
    //double check form of simoid deriv, should it get b.Ia or b.Ib?
    apply_tanhx2_deriv(b.Za.slice(t), Rb.tmp1.slice(t));
    //shout this next line be dot_add?
    mult_add(Rb.tmp1.slice(t), Rb.Za.slice(t), Rb.Zb.slice(t));

    dot(Rb.Ib.slice(t), b.Zb.slice(t), Rb.S.slice(t));
    dot_add(b.Ib.slice(t), Rb.Zb.slice(t), Rb.S.slice(t));

    if (t) {
      dot_add(Rb.S.slice(t - 1), b.Fb.slice(t), Rb.S.slice(t));
      dot_add(b.S.slice(t - 1), Rb.Fb.slice(t), Rb.S.slice(t));
    }
      
    apply_tanhx2_deriv(Rb.S.slice(t), Rb.f_S.slice(t));

    dot_add(Rb.S.slice(t), w.OS, Rb.Oa.slice(t));
    dot_add(b.S.slice(t), v.OS, Rb.Oa.slice(t));
    

    //mult_add(b.S.slice(t), w.OS, b.Oa.slice(t));
    apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
    //copy(b.Oa.slice(t), b.Ob.slice(t));

    //double check form of simoid deriv, should it get b.Ia or b.Ib?
    apply_sigmoid_deriv(b.Oa.slice(t), Rb.tmp1.slice(t));
    //shout this next line be dot_add?
    mult_add(Rb.tmp1.slice(t), Rb.Oa.slice(t), Rb.Ob.slice(t));

    
    dot(b.f_S.slice(t), Rb.Ob.slice(t), Ry.slice(t));
    apply_tanhx2_deriv(b.S.slice(t), Rb.tmp1.slice(t));
    dot_add(Rb.Ob.slice(t), Rb.tmp1.slice(t), Ry.slice(t));
   }
}


//instead of normal deltas buffer, pass in empty Rdeltas buffer, and instead of out_deltas, pass in the Ry value calculated by the Rfwd pass
void lstm_Rbackward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas, MatrixView3DCPU &out_deltas, LstmBuffers &Rb, double lambda, double mu) {

  int end_time = static_cast<int>(b.time - 1);

  copy(out_deltas, d.Hb);
  
  //calculate t+1 values except for end_time+1 
  for(int t(end_time); t >= 0; --t){
    if (t<end_time) { 
    
      mult_add(w.IH.T(), d.Ia.slice(t+1), d.Hb.slice(t));
      mult_add(w.FH.T(), d.Fa.slice(t+1), d.Hb.slice(t));
      mult_add(w.ZH.T(), d.Za.slice(t+1), d.Hb.slice(t));
      mult_add(w.OH.T(), d.Oa.slice(t+1), d.Hb.slice(t));
  
      //! \f$\frac{dE}{dS} += \frac{dE}{dS^{t+1}} * b_F(t+1)\f$
      dot_add(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
      
      //! \f$\frac{dE}{dS} += \frac{dE}{da_I(t+1)} * W_{IS}\f$
      dot_add(d.Ia.slice(t+1), w.IS, d.S.slice(t));

      //! \f$\frac{dE}{dS} += \frac{dE}{da_F(t+1)} * W_{FS}\f$
      dot_add(d.Fa.slice(t+1), w.FS, d.S.slice(t)); 
    }

    //structural damping
    copy(Rb.Hb.slice(t), d.tmp1.slice(t));
    scale_into(d.tmp1.slice(t), lambda*mu);
    add_vector_into(d.tmp1.slice(t), d.Hb.slice(t));
    
    //! \f$\frac{dE}{df_S} += \frac{dE}{dH} * b_O\f$  saves intermediate value, used for dE/dS
    dot(d.Hb.slice(t), b.Ob.slice(t), d.f_S.slice(t));
    
    //OUTPUT GATES DERIVS
    //! \f$\frac{dE}{db_O} = \frac{dE}{dH} * f(s) * f(a_O)\f$
    dot(d.Hb.slice(t), b.f_S.slice(t), d.Ob.slice(t));

    //! \f$\frac{dE}{da_O} = \frac{dE}{db_O} * f'(a_O)\f$
    apply_sigmoid_deriv(b.Ob.slice(t), d.tmp1.slice(t)); //s'(O_a) == s(O_b) * (1 - s(O_b)) 
    dot(d.Ob.slice(t), d.tmp1.slice(t), d.Oa.slice(t));
    
    //State cell derivs
    //! \f$\frac{dE}{dS} += \frac{dE}{df_S} * f'(s)\f$
    apply_tanhx2_deriv(b.S.slice(t), d.tmp1.slice(t));
    dot_add(d.f_S.slice(t), d.tmp1.slice(t), d.S.slice(t));

    //! \f$\frac{dE}{dS} += \frac{dE}{da_O} * W_OS\f$
    dot_add(d.Oa.slice(t), w.OS, d.S.slice(t));
    
    //! CELL ACTIVATION DERIVS
    //! \f$\frac{dE}{db_Z} = \frac{dE}{dS} * b_I\f$
    dot(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
    //! \f$dE/da_Z = dE/db_Z * f'(a_Z)\f$
    apply_tanhx2_deriv(b.Za.slice(t), d.tmp1.slice(t));
    dot(d.Zb.slice(t), d.tmp1.slice(t), d.Za.slice(t));

    //structural damping (this may be in the wrong place, but trying to follow previous version)
    scale_into(d.tmp1.slice(t), lambda*mu);
    dot_add(d.tmp1.slice(t), Rb.Za.slice(t), d.Za.slice(t)); 
    
    //! INPUT GATE DERIVS
    //! \f$\frac{dE}{db_I} = \frac{dE}{dS} * b_Z \f$
    dot(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));

    //! \f$\frac{dE}{da_I} = \frac{dE}{db_I} * f'(a_I) \f$
    //sigmoid_deriv(d.Ib.slice(t), b.Ib.slice(t), d.temp_hidden, d.temp_hidden2, d.Ia.slice(t));
    
    //apply_sigmoid_deriv(b.Ib.slice(t), d.Ia.slice(t));
    apply_sigmoid_deriv(b.Ib.slice(t), d.tmp1.slice(t));
    dot(d.Ib.slice(t), d.tmp1.slice(t), d.Ia.slice(t));
 
   //! FORGET GATE DERIVS
    if (t)
      //! \f$\frac{dE}{db_F} += \frac{dE}{dS} * s(t-1)\f$
      dot(d.S.slice(t), b.S.slice(t - 1), d.Fb.slice(t));
    
    // \f$\frac{dE}{da_F} = \frac{dE}{db_F} * f'(a_F)\f$
    apply_sigmoid_deriv(b.Fb.slice(t), d.tmp1.slice(t));    
    dot(d.Fb.slice(t), d.tmp1.slice(t), d.Fa.slice(t));

    //dE/dx 
    mult(w.IX.T(), d.Ia.slice(t), in_deltas.slice(t));
    mult_add(w.ZX.T(), d.Za.slice(t), in_deltas.slice(t));
    mult_add(w.FX.T(), d.Fa.slice(t), in_deltas.slice(t));    
  
    
  }
}


/*

void lstm_forward(LstmWeights &w, LstmBuffers &b, MatrixView3DCPU &x, MatrixView3DCPU &y) {
void forward_pass(LSTM_Weights<matrix_type> &lstm_weights, matrix3d_type &input_batches)
void f1_pass(LSTM_Weights<matrix_type> &lstm_weights, LSTM_Weights<matrix_type> &v, matrix3d_type &input_batches) {



	      sigmoid_deriv(d_RFa.matrix_from_slice(t), bc::d_Fb.matrix_from_slice(t), bc::d_temp_hidden, bc::d_temp_hidden2, d_RFb.matrix_from_slice(t));
	      sigmoid_deriv(d_RIa.matrix_from_slice(t), bc::d_Ib.matrix_from_slice(t), bc::d_temp_hidden, bc::d_temp_hidden2, d_RIb.matrix_from_slice(t));
	      

	      multiply_vector(bc::d_Cg.matrix_from_slice(t), d_RIb.matrix_from_slice(t), d_RS.matrix_from_slice(t), 1.0);

	      // Weird part
	      //d_RSIb_tmp.matrix_from_slice(t).clear();
	      bc::d_temp_hidden.clear();
	      //d_Cg_tmp_deriv.matrix_from_slice(t).clear();
	      multiply_vector(d_RCa.matrix_from_slice(t), bc::d_Ib.matrix_from_slice(t), bc::d_temp_hidden, 1.0);
	      tanh2_deriv(bc::d_temp_hidden, bc::d_Cg.matrix_from_slice(t), bc::d_temp_hidden2, d_RS.matrix_from_slice(t));
	      //
	      if (t) {
	    	  multiply_vector(d_RFb.matrix_from_slice(t), bc::d_S.matrix_from_slice(t - 1), d_RS.matrix_from_slice(t), 1.0);
	    	  multiply_vector(bc::d_Fb.matrix_from_slice(t), d_RS.matrix_from_slice(t - 1), d_RS.matrix_from_slice(t), 1.0);
	      }

	      multiply_vector(bc::d_S.matrix_from_slice(t), v.d_cO, d_ROa.matrix_from_slice(t), 1.0);
	      multiply_vector(d_RS.matrix_from_slice(t), lstm_weights.d_cO, d_ROa.matrix_from_slice(t), 1.0);

	      add_vector(d_ROa.matrix_from_slice(t), v.d_O_bias);
	      //d_RO_tmp1.matrix_from_slice(t).clear();
	      //d_RO_tmp2.matrix_from_slice(t).clear();
	      sigmoid_deriv(d_ROa.matrix_from_slice(t), bc::d_Ob.matrix_from_slice(t), bc::d_temp_hidden, bc::d_temp_hidden2, d_ROb.matrix_from_slice(t));

	      //d_temp.clear();
	      //apply_tanh(bc::d_Sb.matrix_from_slice(t), d_temp);
	      //multiply_vector(d_ROb.matrix_from_slice(t), d_temp, d_RCb.matrix_from_slice(t));
	      multiply_vector(d_ROb.matrix_from_slice(t), bc::d_Sb.matrix_from_slice(t), d_RCb.matrix_from_slice(t));
	      
	      bc::d_temp_hidden.clear();
	      multiply_vector(bc::d_Ob.matrix_from_slice(t), d_RS.matrix_from_slice(t), bc::d_temp_hidden);

	      //d_ObRS_tmp.clear();
	      //d_RCb_tmp1.matrix_from_slice(t).clear();
	      tanh2_deriv(bc::d_temp_hidden, bc::d_Sb.matrix_from_slice(t), bc::d_temp_hidden2, d_RCb.matrix_from_slice(t));
	    }
	    
	    //output cells
	    multiply(v.d_OT, bc::d_Cb, d_Routput_a);
	    multiply(lstm_weights.d_OT, d_RCb, d_Routput_a);

	    add_vector(d_Routput_a, v.d_T_bias);

	    d_output_tmp.clear(); /////////
	    softmax_deriv(d_Routput_a, bc::d_output_a, d_output_tmp, d_Routput_activations);
	    //if (Stats::instance().epoch == 1 && Stats::instance().first_gv == false) {
	    //  d_RCb.print();
	    //  exit(1);

	}
*/
