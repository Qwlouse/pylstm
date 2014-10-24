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
    S_last(NULL, n_cells/2, n_batches, time), // Cell states from previous layer (NEW!)
    Ia(NULL, n_cells/2, n_batches, time), Ib(NULL, n_cells/2, n_batches, time), //!< Input gate activation
    Fa(NULL, n_cells/2, n_batches, time), Fb(NULL, n_cells/2, n_batches, time), //!< forget gate activation
    Oa(NULL, n_cells/2, n_batches, time), Ob(NULL, n_cells/2, n_batches, time), //!< output gate activation

    Za(NULL, n_cells/2, n_batches, time), Zb(NULL, n_cells/2, n_batches, time), //!< Za =Net Activation, Zb=f(Za)
    S(NULL, n_cells/2, n_batches, time),      //!< Sa =Cell State activations
    f_S(NULL, n_cells/2, n_batches, time),      //!< Sa =Cell State activations
    Hb(NULL, n_cells/2, n_batches, time),     //!< output of LSTM block
    tmp1(NULL, n_cells/2, n_batches, time) // for calculating derivs
{
    add_view("S_last", &S_last);
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
    for (size_t t(1); t < x.n_slices; ++t) {
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
    copy(x.row_slice(0, w.I_bias.n_rows), b.S_last);
    dot_add(b.S_last, b.Fb, b.S);
    f->apply(b.S, b.f_S);

    // Compute outputs
    copy(b.S, y.row_slice(0, w.I_bias.n_rows));
    dot(b.f_S, b.Ob, y.row_slice(w.I_bias.n_rows, y.n_rows));
}


void StaticLstmLayer::backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
    dampened_backward(w, b, d, y, in_deltas, out_deltas, b, 0., 0.);
}


void StaticLstmLayer::gradient(Parameters& w, Parameters& grad, FwdState& b, BwdState& d, Matrix& y, Matrix& x, Matrix& )  {

    //! \f$\frac{dE}{dW_ZX} += \frac{dE}{da_Z} * x(t)\f$
    //! \f$\frac{dE}{dW_FX} += \frac{dE}{da_F} * x(t)\f$
    //! \f$\frac{dE}{dW_IX} += \frac{dE}{da_I} * x(t)\f$
    //! \f$\frac{dE}{dW_OX} += \frac{dE}{da_O} * x(t)\f$
    grad.IX.set_all_elements_to(0.0);
    grad.ZX.set_all_elements_to(0.0);
    grad.FX.set_all_elements_to(0.0);
    grad.OX.set_all_elements_to(0.0);
    for (size_t t(1); t < x.n_slices; ++t) {
        mult_add(d.Za.slice(t), x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t).T(), grad.ZX); //  1.0 / 1.0); //(double) n_time);
        mult_add(d.Fa.slice(t), x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t).T(), grad.FX); // 1.0 / 1.0); //(double) n_time);
        mult_add(d.Ia.slice(t), x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t).T(), grad.IX); //1.0 / 1.0); //(double) n_time);
        mult_add(d.Oa.slice(t), x.row_slice(w.I_bias.n_rows, x.n_rows).slice(t).T(), grad.OX); // 1.0 / 1.0); //(double) n_time);
    }

    squash(d.Ia, grad.I_bias); //, 1.0 / (double) n_time);
    squash(d.Fa, grad.F_bias); //, 1.0 / (double) n_time);
    squash(d.Za, grad.Z_bias); //, 1.0 / (double) n_time);
    squash(d.Oa, grad.O_bias); //, 1.0 / (double)n_time); //, 1.0 / (double) n_time);
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
    
    // Output Gate
    dot(out_deltas.row_slice(w.I_bias.n_rows, y.n_rows), b.f_S, d.Ob);
    apply_sigmoid_deriv(b.Ob, d.tmp1);
    dot(d.Ob, d.tmp1, d.Oa);
    
    // State
    dot(out_deltas.row_slice(w.I_bias.n_rows, y.n_rows), b.Ob, d.f_S);
    f->apply_deriv(b.f_S, d.f_S, d.S);
    // should there be a dot(d.f_S, d.tmp1, d.S); here and d.tmp1 instead of d.S above?
    add_into_b(out_deltas.row_slice(0, w.I_bias.n_rows), d.S);
    
    // Cell
    dot(d.S, b.Ib, d.Zb);
    apply_tanh_deriv(b.Zb, d.tmp1);
    dot(d.Zb, d.tmp1, d.Za);
    
    // Input Gate
    dot(d.S, b.Zb, d.Ib);
    apply_sigmoid_deriv(b.Ib, d.tmp1);
    dot(d.Ib, d.tmp1, d.Ia);
    
    // Forget Gate
    dot(d.S, b.S_last, d.Fb);
    apply_sigmoid_deriv(b.Fb, d.tmp1);
    dot(d.Fb, d.tmp1, d.Fa);
    
    //////////////////////// Delta Clipping ///////////////////////////
    if (delta_range < INFINITY) {
        clip_elements(d.Ia, -delta_range, delta_range);
        clip_elements(d.Oa, -delta_range, delta_range);
        clip_elements(d.Za, -delta_range, delta_range);
        clip_elements(d.Fa, -delta_range, delta_range);
    }
    ///////////////////////////////////////////////////////////////////
    
    // in_deltas
    
    dot_add(d.S, b.Fb, in_deltas.row_slice(0, w.I_bias.n_rows));
    
    for (size_t t(1); t < y.n_slices; ++t) {
        mult_add(w.IX.T(), d.Ia.slice(t), in_deltas.row_slice(w.I_bias.n_rows, y.n_rows).slice(t));
        mult_add(w.OX.T(), d.Oa.slice(t), in_deltas.row_slice(w.I_bias.n_rows, y.n_rows).slice(t));
        mult_add(w.ZX.T(), d.Za.slice(t), in_deltas.row_slice(w.I_bias.n_rows, y.n_rows).slice(t));
        mult_add(w.FX.T(), d.Fa.slice(t), in_deltas.row_slice(w.I_bias.n_rows, y.n_rows).slice(t));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    
}
