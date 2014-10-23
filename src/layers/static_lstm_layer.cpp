#include "static_lstm_layer.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"


StaticLstmLayer::StaticLstmLayer():
	f(&Tanhx2),
	delta_range(INFINITY)
{ }

StaticLstmLayer::StaticLstmLayer(const ActivationFunction* f):
	f(f),
	delta_range(INFINITY)
{ }


// To make a StaticLSTMLayer with n cells, you must specify 2*n as size in Python
StaticLstmLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
    IX(NULL, n_cells/2, n_inputs - (n_cells/2), 1),
    FX(NULL, n_cells/2, n_inputs - (n_cells/2), 1),
    ZX(NULL, n_cells/2, n_inputs - (n_cells/2), 1),
    OX(NULL, n_cells/2, n_inputs - (n_cells/2), 1),
    I_bias(NULL, n_cells/2, 1, 1), F_bias(NULL, n_cells/2, 1, 1), Z_bias(NULL, n_cells/2, 1, 1), O_bias(NULL, n_cells/2, 1, 1)
{

    add_view("IX", &IX);
    add_view("FX", &FX);
    add_view("ZX", &ZX);
    add_view("OX", &OX);
    add_view("I_bias", &I_bias); add_view("F_bias", &F_bias); add_view("Z_bias", &Z_bias); add_view("O_bias", &O_bias);
}


StaticLstmLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    //Views on all activations
    Ia(NULL, n_cells/2, n_batches, time), Ib(NULL, n_cells/2, n_batches, time), //!< Input gate activation
    Fa(NULL, n_cells/2, n_batches, time), Fb(NULL, n_cells/2, n_batches, time), //!< forget gate activation
    Oa(NULL, n_cells/2, n_batches, time), Ob(NULL, n_cells/2, n_batches, time), //!< output gate activation

    Za(NULL, n_cells/2, n_batches, time), Zb(NULL, n_cells/2, n_batches, time), //!< Za =Net Activation, Zb=f(Za)
    S(NULL, n_cells/2, n_batches, time),      //!< Sa =Cell State activations
    f_S(NULL, n_cells/2, n_batches, time),      //!< Sa =Cell State activations
    Hb(NULL, n_cells/2, n_batches, time),     //!< output of LSTM block
    tmp1(NULL, n_cells/2, n_batches, time) // for calculating derivs
{
    add_view("Ia", &Ia); add_view("Ib", &Ib);
    add_view("Fa", &Fa); add_view("Fb", &Fb);
    add_view("Oa", &Oa); add_view("Ob", &Ob);
    add_view("Za", &Za); add_view("Zb", &Zb);
    add_view("S", &S);
    add_view("f_S", &f_S); 
    add_view("Hb", &Hb); 
    add_view("tmp1", &tmp1);
}


StaticLstmLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    //Views on all activations
    Ia(n_cells/2, n_batches, time), Ib(n_cells/2, n_batches, time), //Input gate activation
    Fa(n_cells/2, n_batches, time), Fb(n_cells/2, n_batches, time), //forget gate activation
    Oa(n_cells/2, n_batches, time), Ob(n_cells/2, n_batches, time), //output gate activation

    Za(n_cells/2, n_batches, time), Zb(n_cells/2, n_batches, time), //Net Activation
    S(n_cells/2, n_batches, time), //Cell activations
    f_S(n_cells/2, n_batches, time), //cell state activations
    Hb(n_cells/2, n_batches, time),     //!< output of LSTM block

    tmp1(n_cells/2, n_batches, time) // for calculating derivs
{
    add_view("Ia", &Ia); add_view("Ib", &Ib);
    add_view("Fa", &Fa); add_view("Fb", &Fb);
    add_view("Oa", &Oa); add_view("Ob", &Ob);
    add_view("Za", &Za); add_view("Zb", &Zb);
    add_view("S", &S);
    add_view("f_S", &f_S); 
    add_view("Hb", &Hb); 
    add_view("tmp1", &tmp1);
}


void StaticLstmLayer::forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y, bool) {
  // Convention: first n_cells/2 of the features are cell states from previous layer
  //             rest of the features are outputs from the previous layer

  ASSERT(x.n_rows == w.I_bias.n_rows + w.IX.n_columns);
  // Compute Gates
  for (size_t t(0); t < x.n_slices; ++t) {
    mult(w.IX, x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t), b.Ia.slice(t));
    mult(w.FX, x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t), b.Fa.slice(t));
    mult(w.ZX, x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t), b.Za.slice(t));
    mult(w.OX, x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t), b.Oa.slice(t));
  }

  add_vector_into(w.F_bias, b.Fa);
  add_vector_into(w.I_bias, b.Ia);
  add_vector_into(w.Z_bias, b.Za);
  add_vector_into(w.O_bias, b.Oa);

  apply_sigmoid(b.Fa, b.Fb);
  apply_sigmoid(b.Ia, b.Ib);
  apply_sigmoid(b.Oa, b.Ob);
  apply_tanh(b.Za, b.Zb);
  dot(b.Zb, b.Ib, b.S);

  // Compute Cell
  dot_add(x.row_slice(0, w.I_bias.n_rows), b.Fb, b.S);
  f->apply(b.S, b.f_S);

  // Compute outputs
  copy(b.S, y.row_slice(0, w.I_bias.n_rows));
  dot(b.f_S, b.Ob, y.row_slice(w.I_bias.n_rows, y.n_rows));
}


void StaticLstmLayer::backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
    dampened_backward(w, b, d, y, in_deltas, out_deltas, b, 0., 0.);
}


void StaticLstmLayer::gradient(Parameters&, Parameters& grad, FwdState& b, BwdState& d, Matrix& y, Matrix& x, Matrix& )  {
    size_t n_time = x.n_slices;

//    //! \f$\frac{dE}{dW_ZX} += \frac{dE}{da_Z} * x(t)\f$
//    //! \f$\frac{dE}{dW_FX} += \frac{dE}{da_F} * x(t)\f$
//    //! \f$\frac{dE}{dW_IX} += \frac{dE}{da_I} * x(t)\f$
//    //! \f$\frac{dE}{dW_OX} += \frac{dE}{da_O} * x(t)\f$
//    mult(d.Za.slice(1,d.Za.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.ZX); //  1.0 / 1.0); //(double) n_time);
//    mult(d.Fa.slice(1,d.Fa.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.FX); // 1.0 / 1.0); //(double) n_time);
//    mult(d.Ia.slice(1,d.Ia.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.IX); //1.0 / 1.0); //(double) n_time);
//    mult(d.Oa.slice(1,d.Oa.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.OX); // 1.0 / 1.0); //(double) n_time);
//
//    //! \f$\frac{dE}{dW_ZH} += \frac{dE}{da_Z} * h(t-1)\f$
//    //! \f$\frac{dE}{dW_FH} += \frac{dE}{da_F} * h(t-1)\f$
//    //! \f$\frac{dE}{dW_IH} += \frac{dE}{da_I} * h(t-1)\f$
//    //! \f$\frac{dE}{dW_OH} += \frac{dE}{da_O} * h(t-1)\f$
//    grad.IH.set_all_elements_to(0.0);
//    grad.ZH.set_all_elements_to(0.0);
//    grad.FH.set_all_elements_to(0.0);
//    grad.OH.set_all_elements_to(0.0);
//
//    for (int t = 0; t < n_time - 1; ++t) {
//        mult_add(d.Ia.slice(t+1), y.slice(t).T(), grad.IH); //(double) n_time);
//        mult_add(d.Za.slice(t+1), y.slice(t).T(), grad.ZH); //(double) n_time);
//        mult_add(d.Fa.slice(t+1), y.slice(t).T(), grad.FH); //(double) n_time);
//        mult_add(d.Oa.slice(t+1), y.slice(t).T(), grad.OH); //(double) n_time);
//    }
//
//    //! \f$\frac{dE}{dW_FS} += \frac{dE}{da_F} * s(t-1)\f$
//    //! \f$\frac{dE}{dW_IS} += \frac{dE}{da_I} * s(t-1)\f$
//    if (n_time > 1) {
//        dot_squash(d.Fa.slice(1, n_time), b.S.slice(0, n_time - 1), grad.FS);
//        dot_squash(d.Ia.slice(1, n_time), b.S.slice(0, n_time - 1), grad.IS);
//    }
//
//    //! \f$\frac{dE}{dW_OS} += \frac{dE}{da_O} * s(t)\f$
//    dot_squash(d.Oa, b.S, grad.OS);
//
//    squash(d.Ia, grad.I_bias); //, 1.0 / (double) n_time);
//    squash(d.Fa, grad.F_bias); //, 1.0 / (double) n_time);
//    squash(d.Za, grad.Z_bias); //, 1.0 / (double) n_time);
//    squash(d.Oa, grad.O_bias); //, 1.0 / (double)n_time); //, 1.0 / (double) n_time);
}


void StaticLstmLayer::Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry) {

//  mult(v.IX, x.slice(1,x.n_slices).flatten_time(), Rb.Ia.slice(1,Rb.Ia.n_slices).flatten_time());
//  mult(v.FX, x.slice(1,x.n_slices).flatten_time(), Rb.Fa.slice(1,Rb.Fa.n_slices).flatten_time());
//  mult(v.ZX, x.slice(1,x.n_slices).flatten_time(), Rb.Za.slice(1,Rb.Za.n_slices).flatten_time());
//  mult(v.OX, x.slice(1,x.n_slices).flatten_time(), Rb.Oa.slice(1,Rb.Oa.n_slices).flatten_time());
//
//  mult_add(w.IX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Ia.slice(1,Rb.Ia.n_slices).flatten_time());
//  mult_add(w.FX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Fa.slice(1,Rb.Fa.n_slices).flatten_time());
//  mult_add(w.ZX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Za.slice(1,Rb.Za.n_slices).flatten_time());
//  mult_add(w.OX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Oa.slice(1,Rb.Oa.n_slices).flatten_time());
//
//
//  for (size_t t(1); t < x.n_slices; ++t) {
//
//    mult_add(w.IH, Ry.slice(t - 1), Rb.Ia.slice(t));
//    mult_add(w.FH, Ry.slice(t - 1), Rb.Fa.slice(t));
//    mult_add(w.ZH, Ry.slice(t - 1), Rb.Za.slice(t));
//    mult_add(w.OH, Ry.slice(t - 1), Rb.Oa.slice(t));
//
//    mult_add(v.IH, y.slice(t - 1), Rb.Ia.slice(t));
//    mult_add(v.FH, y.slice(t - 1), Rb.Fa.slice(t));
//    mult_add(v.ZH, y.slice(t - 1), Rb.Za.slice(t));
//    mult_add(v.OH, y.slice(t - 1), Rb.Oa.slice(t));
//
//    dot_add(Rb.S.slice(t - 1), w.IS, Rb.Ia.slice(t));
//    dot_add(Rb.S.slice(t - 1), w.FS, Rb.Fa.slice(t));
//
//    dot_add(b.S.slice(t - 1), v.IS, Rb.Ia.slice(t));
//    dot_add(b.S.slice(t - 1), v.FS, Rb.Fa.slice(t));
//
//
//    add_vector_into(v.I_bias, Rb.Ia.slice(t));
//    add_vector_into(v.F_bias, Rb.Fa.slice(t));
//    add_vector_into(v.Z_bias, Rb.Za.slice(t));
//    add_vector_into(v.O_bias, Rb.Oa.slice(t));
//
//    apply_sigmoid_deriv(b.Ib.slice(t), Rb.tmp1.slice(t));
//    dot(Rb.tmp1.slice(t), Rb.Ia.slice(t), Rb.Ib.slice(t));
//
//    apply_sigmoid_deriv(b.Fb.slice(t), Rb.tmp1.slice(t));
//    dot(Rb.tmp1.slice(t), Rb.Fa.slice(t), Rb.Fb.slice(t));
//
//    apply_tanh_deriv(b.Zb.slice(t), Rb.tmp1.slice(t));
//    dot(Rb.tmp1.slice(t), Rb.Za.slice(t), Rb.Zb.slice(t));
//
//
//    dot(Rb.Ib.slice(t), b.Zb.slice(t), Rb.S.slice(t));
//    dot_add(b.Ib.slice(t), Rb.Zb.slice(t), Rb.S.slice(t));
//
//    dot_add(Rb.S.slice(t - 1), b.Fb.slice(t), Rb.S.slice(t));
//    dot_add(b.S.slice(t - 1), Rb.Fb.slice(t), Rb.S.slice(t));
//
//    dot_add(Rb.S.slice(t), w.OS, Rb.Oa.slice(t));
//    dot_add(b.S.slice(t), v.OS, Rb.Oa.slice(t));
//
//    apply_sigmoid_deriv(b.Ob.slice(t), Rb.tmp1.slice(t));
//    dot(Rb.tmp1.slice(t), Rb.Oa.slice(t), Rb.Ob.slice(t));
//
//    f->apply_deriv(b.f_S.slice(t), Rb.S.slice(t), Rb.tmp1.slice(t));
//    dot(Rb.tmp1.slice(t), b.Ob.slice(t), Ry.slice(t));
//    dot_add(Rb.Ob.slice(t), b.f_S.slice(t), Ry.slice(t));
//  }
}


//instead of normal deltas buffer, pass in empty Rdeltas buffer, and instead of out_deltas, pass in the Ry value calculated by the Rfwd pass
void StaticLstmLayer::dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu) {
//  int end_time = static_cast<int>(y.n_slices - 1);
//  copy(out_deltas, d.Hb);
//
//  //calculate t+1 values except for end_time+1
//  for(int t(end_time); t >= 0; --t){
//      if (t<end_time) {
//          mult_add(w.IH.T(), d.Ia.slice(t+1), d.Hb.slice(t));
//          mult_add(w.FH.T(), d.Fa.slice(t+1), d.Hb.slice(t));
//          mult_add(w.ZH.T(), d.Za.slice(t+1), d.Hb.slice(t));
//          mult_add(w.OH.T(), d.Oa.slice(t+1), d.Hb.slice(t));
//
//          //! \f$\frac{dE}{dS} += \frac{dE}{dS^{t+1}} * b_F(t+1)\f$
//          dot(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
//
//          //! \f$\frac{dE}{dS} += \frac{dE}{da_I(t+1)} * W_{IS}\f$
//          dot_add(d.Ia.slice(t+1), w.IS, d.S.slice(t));
//
//          //! \f$\frac{dE}{dS} += \frac{dE}{da_F(t+1)} * W_{FS}\f$
//          dot_add(d.Fa.slice(t+1), w.FS, d.S.slice(t));
//      }
//
//      // STRUCTURAL DAMPING JUST ON HIDDEN BLOCK --- ASSUMING NO NONLINEARITY
//      copy(Rb.Hb.slice(t), d.tmp1.slice(t));
//      scale_into(d.tmp1.slice(t), lambda*mu);
//      add_into_b(d.tmp1.slice(t), d.Hb.slice(t));
//
//
//      //! \f$\frac{dE}{df_S} += \frac{dE}{dH} * b_O\f$  THIS IS WEIRD, IT GOES WITH NEXT LINE ??!?!
//      dot(d.Hb.slice(t), b.Ob.slice(t), d.f_S.slice(t));
//
//      //OUTPUT GATES DERIVS
//      //! \f$\frac{dE}{db_O} = \frac{dE}{dH} * f(s) * f(a_O)\f$
//      dot(d.Hb.slice(t), b.f_S.slice(t), d.Ob.slice(t));
//
//      //! \f$\frac{dE}{da_O} = \frac{dE}{db_O} * f'(a_O)\f$
//      apply_sigmoid_deriv(b.Ob.slice(t), d.tmp1.slice(t)); //s'(O_a) == s(O_b) * (1 - s(O_b))
//      dot(d.Ob.slice(t), d.tmp1.slice(t), d.Oa.slice(t));
//
//      //! \f$\frac{dE}{dS} += \frac{dE}{df_S} * f'(s)\f$
//      f->apply_deriv(b.f_S.slice(t), d.f_S.slice(t), d.tmp1.slice(t));
//      //dot_add(d.f_S.slice(t), d.tmp1.slice(t), d.S.slice(t));
//
//      if(t<end_time)
//          {add_into_b(d.tmp1.slice(t), d.S.slice(t));}
//      else
//          {copy(d.tmp1.slice(t), d.S.slice(t));}
//
//      //! \f$\frac{dE}{dS} += \frac{dE}{da_O} * W_OS\f$
//      dot_add(d.Oa.slice(t), w.OS, d.S.slice(t));
//
//      // STRUCTURAL DAMPING JUST ON STATE CELL
//      copy(Rb.f_S.slice(t), d.tmp1.slice(t));
//      scale_into(d.tmp1.slice(t), lambda*mu);
//      add_into_b(d.tmp1.slice(t), d.S.slice(t));
//
//      //! CELL ACTIVATION DERIVS
//      //! \f$\frac{dE}{db_Z} = \frac{dE}{dS} * b_I\f$
//      dot(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
//      //! \f$dE/da_Z = dE/db_Z * f'(a_Z)\f$
//      apply(b.Zb.slice(t), d.tmp1.slice(t), &tanh_deriv);
//      dot(d.Zb.slice(t), d.tmp1.slice(t), d.Za.slice(t));
//
//
//      //! INPUT GATE DERIVS
//      //! \f$\frac{dE}{db_I} = \frac{dE}{dS} * b_Z \f$
//      dot(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));
//
//      //! \f$\frac{dE}{da_I} = \frac{dE}{db_I} * f'(a_I) \f$
//      //sigmoid_deriv(d.Ib.slice(t), b.Ib.slice(t), d.temp_hidden, d.temp_hidden2, d.Ia.slice(t));
//
//      //apply_sigmoid_deriv(b.Ib.slice(t), d.Ia.slice(t));
//      apply_sigmoid_deriv(b.Ib.slice(t), d.tmp1.slice(t));
//      dot(d.Ib.slice(t), d.tmp1.slice(t), d.Ia.slice(t));
//
//      //! INPUT GATE DERIVS
//      //! \f$\frac{dE}{db_I} = \frac{dE}{dS} * b_Z \f$
//      dot(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));
//
//      //! \f$\frac{dE}{da_I} = \frac{dE}{db_I} * f'(a_I) \f$
//      //sigmoid_deriv(d.Ib.slice(t), b.Ib.slice(t), d.temp_hidden, d.temp_hidden2, d.Ia.slice(t));
//
//      //apply_sigmoid_deriv(b.Ib.slice(t), d.Ia.slice(t));
//      apply_sigmoid_deriv(b.Ib.slice(t), d.tmp1.slice(t));
//      dot(d.Ib.slice(t), d.tmp1.slice(t), d.Ia.slice(t));
//
//     //! FORGET GATE DERIVS
//      if (t) {
//          //! \f$\frac{dE}{db_F} += \frac{dE}{dS} * s(t-1)\f$
//          dot(d.S.slice(t), b.S.slice(t - 1), d.Fb.slice(t));
//      } else {
//          d.Fb.slice(t).set_all_elements_to(0.0);
//      }
//
//      // \f$\frac{dE}{da_F} = \frac{dE}{db_F} * f'(a_F)\f$
//      apply_sigmoid_deriv(b.Fb.slice(t), d.tmp1.slice(t));
//      dot(d.Fb.slice(t), d.tmp1.slice(t), d.Fa.slice(t));
//
//
//      //////////////////////// Alex Graves Delta Clipping ///////////////////////////
//      if (delta_range < INFINITY) {
//          clip_elements(d.Ia.slice(t), -delta_range, delta_range);
//          clip_elements(d.Oa.slice(t), -delta_range, delta_range);
//          clip_elements(d.Za.slice(t), -delta_range, delta_range);
//          clip_elements(d.Fa.slice(t), -delta_range, delta_range);
//      }
//      //////////////////////////////////////////////////////////////////////////
//
//      //dE/dx
//      // if(t) because of 1-indexing
//      if(t) {
//	      mult_add(w.IX.T(), d.Ia.slice(t), in_deltas.slice(t));
//	      mult_add(w.OX.T(), d.Oa.slice(t), in_deltas.slice(t));
//	      mult_add(w.ZX.T(), d.Za.slice(t), in_deltas.slice(t));
//	      mult_add(w.FX.T(), d.Fa.slice(t), in_deltas.slice(t));
//      }
//
//  }
}
