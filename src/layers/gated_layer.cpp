#include "gated_layer.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"


GatedLayer::GatedLayer():
	f(&Tanhx2),
	delta_range(INFINITY),
	input_act_func(&Sigmoid)
{ }

GatedLayer::GatedLayer(const ActivationFunction* f):
	f(f),
	delta_range(INFINITY),
	input_act_func(&Sigmoid)
{ }


// To make a StaticLSTMLayer with n cells, you must specify 2*n as size in Python
GatedLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
    IX(NULL, n_cells, n_inputs, 1),
    ZX(NULL, n_cells, n_inputs, 1),
    I_bias(NULL, n_cells, 1, 1), Z_bias(NULL, n_cells, 1, 1)
{
    ASSERT (n_cells == n_inputs);
    add_view("IX", &IX);
    add_view("ZX", &ZX);
    add_view("I_bias", &I_bias);
    add_view("Z_bias", &Z_bias);
}


GatedLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    //Views on all activations
    S_last(NULL, n_cells, n_batches, time), // Cell states from previous layer (NEW!)
    Ia(NULL, n_cells, n_batches, time), Ib(NULL, n_cells, n_batches, time), //!< Input gate activation
    Za(NULL, n_cells, n_batches, time), Zb(NULL, n_cells, n_batches, time), //!< Za =Net Activation, Zb=f(Za)
    Fb(NULL, n_cells, n_batches, time) // for calculating derivs
{
    add_view("S_last", &S_last);
    add_view("Ia", &Ia); add_view("Ib", &Ib);
    add_view("Za", &Za); add_view("Zb", &Zb);
    add_view("Fb", &Fb);
}


GatedLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    //Views on all activations
    Ia(n_cells, n_batches, time), Ib(n_cells, n_batches, time), //Input gate activation
    Za(n_cells, n_batches, time), Zb(n_cells, n_batches, time), //Net Activation
    tmp1(n_cells, n_batches, time) // for calculating derivs
{
    add_view("Ia", &Ia); add_view("Ib", &Ib);
    add_view("Za", &Za); add_view("Zb", &Zb);
    add_view("tmp1", &tmp1);
}


void GatedLayer::forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y, bool) {
    // Compute Gates
    mult(w.IX, x.slice(1, x.n_slices).flatten_time(), b.Ia.slice(1, b.Ia.n_slices).flatten_time());
    mult(w.ZX, x.slice(1, x.n_slices).flatten_time(), b.Za.slice(1, b.Za.n_slices).flatten_time());
    add_vector_into(w.I_bias, b.Ia);
    add_vector_into(w.Z_bias, b.Za);

    // Computation
    input_act_func->apply(b.Ia, b.Ib);
    f->apply(b.Za, b.Zb);
    dot(b.Zb, b.Ib, y);

    // Memory
    copy(x, b.S_last);
    apply(b.Ia, b.Fb, &one_minus_sigmoid);
    dot_add(b.Fb, b.S_last.slice(1, b.S_last.n_slices).flatten_time(), y.flatten_time());
}


void GatedLayer::backward(Parameters& w, FwdState& b, BwdState& d, Matrix&, Matrix& in_deltas, Matrix& out_deltas) {

    dot_add(out_deltas.slice(1, out_deltas.n_slices), b.Fb, in_deltas.slice(1, in_deltas.n_slices));

    subtract(b.Zb, b.S_last, d.tmp1);
    dot(out_deltas, d.tmp1, d.Ib);
    input_act_func->apply_deriv(b.Ia, d.Ib, d.Ia);

    dot(out_deltas, b.Ib, d.Zb);
    f->apply_deriv(b.Zb, d.Zb, d.Za);

    //////////////////////// Delta Clipping ///////////////////////////
    if (delta_range < INFINITY) {
        clip_elements(d.Ia, -delta_range, delta_range);
        clip_elements(d.Za, -delta_range, delta_range);
    }
    ///////////////////////////////////////////////////////////////////

    mult_add(w.IX.T(), d.Ia.slice(1, d.Ia.n_slices).flatten_time(), in_deltas.slice(1, in_deltas.n_slices).flatten_time());
    mult_add(w.ZX.T(), d.Za.slice(1, d.Za.n_slices).flatten_time(), in_deltas.slice(1, in_deltas.n_slices).flatten_time());

}


void GatedLayer::gradient(Parameters&, Parameters& grad, FwdState&, BwdState& d, Matrix&, Matrix& x, Matrix& )  {

    //! \f$\frac{dE}{dW_ZX} += \frac{dE}{da_Z} * x(t)\f$
    //! \f$\frac{dE}{dW_FX} += \frac{dE}{da_F} * x(t)\f$
    //! \f$\frac{dE}{dW_IX} += \frac{dE}{da_I} * x(t)\f$
    //! \f$\frac{dE}{dW_OX} += \frac{dE}{da_O} * x(t)\f$
    grad.IX.set_all_elements_to(0.0);
    grad.ZX.set_all_elements_to(0.0);

    mult_add(d.Ia.slice(1, d.Ia.n_slices), x.slice(1, x.n_slices).T(), grad.IX); //1.0 / 1.0); //(double) n_time);
    mult_add(d.Za.slice(1, d.Za.n_slices), x.slice(1, x.n_slices).T(), grad.ZX); //  1.0 / 1.0); //(double) n_time);

    squash(d.Ia.slice(1, d.Ia.n_slices), grad.I_bias); //, 1.0 / (double) n_time);
    squash(d.Za.slice(1, d.Za.n_slices), grad.Z_bias); //, 1.0 / (double) n_time);
}


void GatedLayer::Rpass(Parameters &, Parameters &,  FwdState &, FwdState &, Matrix &, Matrix &, Matrix&, Matrix &) {

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
void GatedLayer::dampened_backward(Parameters &, FwdState &, BwdState &, Matrix&, Matrix &, Matrix &, FwdState &, double, double) {

}
