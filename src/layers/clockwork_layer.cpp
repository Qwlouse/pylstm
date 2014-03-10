#include "clockwork_layer.h"

#include <cmath>
#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

using std::vector;

ClockworkLayer::ClockworkLayer():
	f(&Sigmoid)
{ }

ClockworkLayer::ClockworkLayer(const ActivationFunction* f):
	f(f)
{ }

ClockworkLayer::~ClockworkLayer()
{
}

ClockworkLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
  HX(NULL, n_cells, n_inputs, 1),
  HR(NULL, n_cells, n_cells, 1),
  Timing(NULL, n_cells, 1, 1),
  H_bias(NULL, n_cells, 1, 1)
{
    add_view("HX", &HX);
    add_view("HR", &HR);
    add_view("Timing", &Timing);
    add_view("H_bias", &H_bias);
}

////////////////////// Fwd Buffer /////////////////////////////////////////////

ClockworkLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
  Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

ClockworkLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    Ha(NULL, n_cells, n_batches, time),
    Hb(NULL, n_cells, n_batches, time),
    tmp(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
	add_view("Hb", &Hb);
	add_view("tmp", &tmp);
}

////////////////////// Helpers /////////////////////////////////////////////
void undo_inactive_nodes(const Matrix& y_old, Matrix y, const Matrix& D, const int t) {
    for (int row = 0; row < D.n_rows; ++row) {
        if (fmod(t, D.get(row, 0, 0)) != 0) {
            for (int col = 0; col < y.n_columns; ++col) {
                y.get(row, col, 0) = y_old.get(row, col, 0);
            }
        }
    }
}

void set_inactive_nodes_to_zero(Matrix y, const Matrix& D, const int t) {
    for (int row = 0; row < D.n_rows; ++row) {
        if (fmod(t, D.get(row, 0, 0)) != 0) {
            for (int col = 0; col < y.n_columns; ++col) {
                y.get(row, col, 0) = 0.0;
            }
        }
    }
}

void copy_errors_of_inactive_nodes(Matrix y_old, const Matrix& y, const Matrix& D, const int t) {
    for (int row = 0; row < D.n_rows; ++row) {
        if (fmod(t, D.get(row, 0, 0)) != 0) {
            for (int col = 0; col < y.n_columns; ++col) {
                y_old.get(row, col, 0) += y.get(row, col, 0);
            }
        }
    }
}


////////////////////// Methods /////////////////////////////////////////////
void ClockworkLayer::forward(ClockworkLayer::Parameters& w, ClockworkLayer::FwdState& b, Matrix& x, Matrix& y, bool) {
    size_t n_slices = x.n_slices;
    mult(w.HX, x.slice(1,x.n_slices).flatten_time(), b.Ha.slice(1,b.Ha.n_slices).flatten_time());
    for (int t = 1; t < n_slices; ++t) {
        mult_add(w.HR, y.slice(t-1), b.Ha.slice(t));

        add_vector_into(w.H_bias, b.Ha.slice(t));
        f->apply(b.Ha.slice(t), y.slice(t));

        undo_inactive_nodes(y.slice(t-1), y.slice(t), w.Timing, t);
    }
}

void ClockworkLayer::backward(ClockworkLayer::Parameters& w, ClockworkLayer::FwdState&, ClockworkLayer::BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
        size_t n_slices = y.n_slices;
    f->apply_deriv(y.slice(n_slices-1), out_deltas.slice(n_slices-1), d.Ha.slice(n_slices-1));
    copy(out_deltas.slice(n_slices-1), d.Hb.slice(n_slices-1));
    set_inactive_nodes_to_zero(d.Ha.slice(n_slices-1), w.Timing, n_slices-1);
    for (int t = static_cast<int>(n_slices - 2); t >= 0; --t) {
        copy(out_deltas.slice(t), d.Hb.slice(t));
        copy_errors_of_inactive_nodes(d.Hb.slice(t), d.Hb.slice(t+1), w.Timing, t+1);
        mult_add(w.HR.T(), d.Ha.slice(t+1), d.Hb.slice(t));
        f->apply_deriv(y.slice(t), d.Hb.slice(t), d.Ha.slice(t));
        set_inactive_nodes_to_zero(d.Ha.slice(t), w.Timing, t);
    }

    mult_add(w.HX.T(), d.Ha.slice(1,d.Ha.n_slices).flatten_time(), in_deltas.slice(1,in_deltas.n_slices).flatten_time());
}

void ClockworkLayer::gradient(ClockworkLayer::Parameters&, ClockworkLayer::Parameters& grad, ClockworkLayer::FwdState& , ClockworkLayer::BwdState& d, Matrix& y, Matrix& x, Matrix&) {
    size_t n_slices = x.n_slices;
    //mult_add(d.Ha.slice(0), x.slice(0).T(), grad.HX);
    for (int t = 1; t < n_slices; ++t) {
        mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
        mult_add(d.Ha.slice(t), y.slice(t-1).T(), grad.HR);
    }
    squash(d.Ha, grad.H_bias);
    grad.Timing.set_all_elements_to(0.0);
}

void ClockworkLayer::Rpass(Parameters& w, Parameters& v,  FwdState&, FwdState& Rb, Matrix& x, Matrix& y, Matrix& Rx, Matrix& Ry)
{
    // NOT IMPLEMENTED YET
    size_t n_slices = x.n_slices;
    mult(v.HX, x.slice(1,x.n_slices).flatten_time(), Rb.Ha.slice(1,Rb.Ha.n_slices).flatten_time());
    mult_add(w.HX, Rx.slice(1,x.n_slices).flatten_time(), Rb.Ha.slice(1,Rb.Ha.n_slices).flatten_time());
    for (int t = 1; t < n_slices; ++t) {
      mult_add(v.HR, y.slice(t-1), Rb.Ha.slice(t));
      mult_add(w.HR, Ry.slice(t-1), Rb.Ha.slice(t));
     
      add_vector_into(v.H_bias, Rb.Ha.slice(t));
      f->apply_deriv(y.slice(t), Rb.Ha.slice(t), Ry.slice(t));
    }
    // NOT IMPLEMENTED YET
}

void ClockworkLayer::dampened_backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState&, double, double)
{
    backward(w, b, d, y, in_deltas, out_deltas);
}

