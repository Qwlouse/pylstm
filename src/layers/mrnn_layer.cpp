#include "mrnn_layer.h"

#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

using std::vector;

MrnnLayer::MrnnLayer():
	f(&Sigmoid)
{ }

MrnnLayer::MrnnLayer(const ActivationFunction* f):
	f(f)
{ }

MrnnLayer::~MrnnLayer()
{
}

MrnnLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
  HX(NULL, n_cells, n_inputs, 1),
  FX(NULL, n_cells, n_inputs, 1),
  FH(NULL, n_cells, n_cells, 1),
  HF(NULL, n_cells, n_cells, 1),
  H_bias(NULL, n_cells, 1, 1)
{
    add_view("HX", &HX);
    add_view("FX", &FX);
    add_view("FH", &FH);
    add_view("HF", &HF);
    add_view("H_bias", &H_bias);
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

MrnnLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
  Ha(NULL, n_cells, n_batches, time),
  F1(NULL, n_cells, n_batches, time),
  F2(NULL, n_cells, n_batches, time),
  Fa(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	add_view("F1", &F1);
	add_view("F2", &F2);
	add_view("Fa", &Fa);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

MrnnLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
  Ha(NULL, n_cells, n_batches, time),
  Hb(NULL, n_cells, n_batches, time),
  F1(NULL, n_cells, n_batches, time),
  F2(NULL, n_cells, n_batches, time),
  Fa(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	add_view("Hb", &Hb);
	add_view("F1", &F1);
	add_view("F2", &F2);
	add_view("Fa", &Fa);
}

////////////////////// Methods /////////////////////////////////////////////
void MrnnLayer::forward(MrnnLayer::Parameters& w, MrnnLayer::FwdState& b, Matrix& x, Matrix& y) {
    size_t n_slices = x.n_slices;
    mult(w.HX, x.slice(1,x.n_slices).flatten_time(), b.Ha.slice(1,b.Ha.n_slices).flatten_time());
    mult(w.FX, x.slice(1,x.n_slices).flatten_time(), b.F1.slice(1,b.F1.n_slices).flatten_time());

    for (int t = 1; t < n_slices; ++t) {
      mult(w.FH, y.slice(t-1), b.F2.slice(t));
      dot(b.F1.slice(t), b.F2.slice(t), b.Fa.slice(t));
      mult_add(w.HF, b.Fa.slice(t), b.Ha.slice(t));
     
      add_vector_into(w.H_bias, b.Ha.slice(t));
      f->apply(b.Ha.slice(t), y.slice(t));
    }
}


void MrnnLayer::backward(MrnnLayer::Parameters& w, MrnnLayer::FwdState& b, MrnnLayer::BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
    size_t n_slices = y.n_slices;
    f->apply_deriv(y.slice(n_slices-1), out_deltas.slice(n_slices-1), d.Ha.slice(n_slices-1));
    mult(w.HF.T(), d.Ha.slice(n_slices-1), d.Fa.slice(n_slices-1));
    dot(d.Fa.slice(n_slices-1), b.F1.slice(n_slices-1), d.F2.slice(n_slices-1));
    dot(d.Fa.slice(n_slices-1), b.F2.slice(n_slices-1), d.F1.slice(n_slices-1));
    for (int t = static_cast<int>(n_slices - 2); t >= 0; --t) {
        copy(out_deltas.slice(t), d.Hb.slice(t));
        mult_add(w.FH.T(), d.F2.slice(t+1), d.Hb.slice(t));
        f->apply_deriv(y.slice(t), d.Hb.slice(t), d.Ha.slice(t));

        mult(w.HF.T(), d.Ha.slice(t), d.Fa.slice(t));
        dot(d.Fa.slice(t), b.F1.slice(t), d.F2.slice(t));
    }
    dot(d.Fa, b.F2, d.F1);
    mult_add(w.HX.T(), d.Ha.slice(1,d.Ha.n_slices).flatten_time(), in_deltas.slice(1,in_deltas.n_slices).flatten_time());
    mult_add(w.FX.T(), d.F1.slice(1,d.F1.n_slices).flatten_time(), in_deltas.slice(1,in_deltas.n_slices).flatten_time());
}



void MrnnLayer::gradient(MrnnLayer::Parameters&, MrnnLayer::Parameters& grad, MrnnLayer::FwdState& b, MrnnLayer::BwdState& d, Matrix& y, Matrix& x, Matrix&) {
    size_t n_slices = x.n_slices;
    for (int t = 0; t < n_slices; ++t) {
      if(t) mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
      mult_add(d.Ha.slice(t), b.Fa.slice(t).T(), grad.HF);
      if(t)  mult_add(d.F1.slice(t), x.slice(t).T(), grad.FX);
      if (t) {
	mult_add(d.F2.slice(t), y.slice(t-1).T(), grad.FH);
      }
    }
    
    squash(d.Ha, grad.H_bias);
}

void MrnnLayer::Rpass(Parameters& w, Parameters& v,  FwdState& b, FwdState& Rb, Matrix& x, Matrix& y, Matrix& Rx, Matrix& Ry)
{
    size_t n_slices = x.n_slices;
    mult(    v.HX,  x.slice(1,x.n_slices).flatten_time(),  Rb.Ha.slice(1,Rb.Ha.n_slices).flatten_time());
    mult_add(w.HX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Ha.slice(1,Rb.Ha.n_slices).flatten_time());

    mult(    v.FX,  x.slice(1,x.n_slices).flatten_time(),  Rb.F1.slice(1,Rb.F1.n_slices).flatten_time());
    mult_add(w.FX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.F1.slice(1,Rb.F1.n_slices).flatten_time());

    for (int t = 0; t < n_slices; ++t) {
      mult(    v.FH,  y.slice(t-1), Rb.F2.slice(t));
      mult_add(w.FH, Ry.slice(t-1), Rb.F2.slice(t));
      
      dot(   Rb.F1.slice(t),  b.F2.slice(t), Rb.Fa.slice(t));
      dot_add(b.F1.slice(t), Rb.F2.slice(t), Rb.Fa.slice(t));
      
      mult_add(v.HF,  b.Fa.slice(t), Rb.Ha.slice(t));
      mult_add(w.HF, Rb.Fa.slice(t), Rb.Ha.slice(t));

      add_vector_into(v.H_bias, Rb.Ha.slice(t));
      f->apply_deriv(y.slice(t), Rb.Ha.slice(t), Ry.slice(t));
    }

}

void MrnnLayer::dampened_backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState&, double, double)
{
    backward(w, b, d, y, in_deltas, out_deltas);
}

