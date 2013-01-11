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
  ZX(n_cells, n_cells), ZH(n_cells, n_cells),
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




