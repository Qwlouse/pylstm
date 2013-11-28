#include "arnn_layer.h"

#include <cmath>
#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"

using std::vector;

ArnnLayer::ArnnLayer():
	f(&Sigmoid)
{ }

ArnnLayer::ArnnLayer(const ActivationFunction* f):
	f(f)
{ }

ArnnLayer::~ArnnLayer()
{
}

ArnnLayer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
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

ArnnLayer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
  Ha(NULL, n_cells, n_batches, time)
{
	add_view("Ha", &Ha);
}

////////////////////// Bwd Buffer /////////////////////////////////////////////

ArnnLayer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
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
void ArnnLayer::forward(ArnnLayer::Parameters& w, ArnnLayer::FwdState& b, Matrix& x, Matrix& y) {
    size_t n_slices = x.n_slices;
    mult(w.HX, x.flatten_time(), b.Ha.flatten_time());
    for (int t = 0; t < n_slices; ++t) {
        if (t) {
            mult_add(w.HR, y.slice(t-1), b.Ha.slice(t));
        }
        add_vector_into(w.H_bias, b.Ha.slice(t));
        f->apply(b.Ha.slice(t), y.slice(t));
        if (t) { // undo changes to non-active units
            undo_inactive_nodes(y.slice(t-1), y.slice(t), w.Timing, t);
        }
    }
}

void ArnnLayer::backward(ArnnLayer::Parameters& w, ArnnLayer::FwdState&, ArnnLayer::BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
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

    mult_add(w.HX.T(), d.Ha.flatten_time(), in_deltas.flatten_time());
}

void ArnnLayer::gradient(ArnnLayer::Parameters&, ArnnLayer::Parameters& grad, ArnnLayer::FwdState& , ArnnLayer::BwdState& d, Matrix& y, Matrix& x, Matrix&) {
    size_t n_slices = x.n_slices;
    mult_add(d.Ha.slice(0), x.slice(0).T(), grad.HX);
    for (int t = 1; t < n_slices; ++t) {
        mult_add(d.Ha.slice(t), x.slice(t).T(), grad.HX);
        mult_add(d.Ha.slice(t), y.slice(t-1).T(), grad.HR);
    }
    squash(d.Ha, grad.H_bias);
    grad.Timing.set_all_elements_to(0.0);
}

void ArnnLayer::Rpass(Parameters& w, Parameters& v,  FwdState&, FwdState& Rb, Matrix& x, Matrix& y, Matrix& Rx, Matrix& Ry)
{
    // NOT IMPLEMENTED YET
    size_t n_slices = x.n_slices;
    mult(v.HX, x.flatten_time(), Rb.Ha.flatten_time());
    mult_add(w.HX, Rx.flatten_time(), Rb.Ha.flatten_time());
    for (int t = 0; t < n_slices; ++t) {
      if (t) {
        mult_add(v.HR, y.slice(t-1), Rb.Ha.slice(t));
        mult_add(w.HR, Ry.slice(t-1), Rb.Ha.slice(t));
      }
      add_vector_into(v.H_bias, Rb.Ha.slice(t));
      f->apply_deriv(y.slice(t), Rb.Ha.slice(t), Ry.slice(t));
    }
    // NOT IMPLEMENTED YET
}

void ArnnLayer::dampened_backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas, FwdState&, double, double)
{
    backward(w, b, d, y, in_deltas, out_deltas);
}

