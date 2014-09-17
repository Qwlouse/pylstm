#include "lstm97_layer.h"

#include <iostream>
#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"


Lstm97Layer::Lstm97Layer():
	f(&Tanhx2),
    full_gradient(true),
    peephole_connections(true),
    forget_gate(true),
    output_gate(true),
    input_gate(true),
    gate_recurrence(true),
    use_bias(true)
{ }

Lstm97Layer::Lstm97Layer(const ActivationFunction* f):
	f(f),
    full_gradient(true),
    peephole_connections(true),
    forget_gate(true),
    output_gate(true),
    input_gate(true),
    gate_recurrence(true),
    use_bias(true)
{ }


Lstm97Layer::Parameters::Parameters(size_t n_inputs, size_t n_cells) :
    IX(NULL, n_cells, n_inputs, 1), IH(NULL, n_cells, n_cells, 1), IS(NULL, 1, n_cells, 1),
    II(NULL, n_cells, n_cells, 1), IF(NULL, n_cells, n_cells, 1), IO(NULL, n_cells, n_cells, 1),
    
    FX(NULL, n_cells, n_inputs, 1), FH(NULL, n_cells, n_cells, 1), FS(NULL, 1, n_cells, 1),
    FI(NULL, n_cells, n_cells, 1), FF(NULL, n_cells, n_cells, 1), FO(NULL, n_cells, n_cells, 1),

    ZX(NULL, n_cells, n_inputs, 1), ZH(NULL, n_cells, n_cells, 1),

    OX(NULL, n_cells, n_inputs, 1), OH(NULL, n_cells, n_cells, 1), OS(NULL, 1, n_cells, 1),
    OI(NULL, n_cells, n_cells, 1),  OF(NULL, n_cells, n_cells, 1), OO(NULL, n_cells, n_cells, 1),

    I_bias(NULL, n_cells, 1, 1), F_bias(NULL, n_cells, 1, 1), Z_bias(NULL, n_cells, 1, 1), O_bias(NULL, n_cells, 1, 1)
{

    add_view("IX", &IX); add_view("IH", &IH); add_view("IS", &IS);
    add_view("II", &II); add_view("IF", &IF); add_view("IO", &IO);

    add_view("FX", &FX); add_view("FH", &FH); add_view("FS", &FS);
    add_view("FI", &FI); add_view("FF", &FF); add_view("FO", &FO);

    add_view("ZX", &ZX); add_view("ZH", &ZH);

    add_view("OX", &OX); add_view("OH", &OH); add_view("OS", &OS);
    add_view("OI", &OI); add_view("OF", &OF); add_view("OO", &OO);

    add_view("I_bias", &I_bias); add_view("F_bias", &F_bias); add_view("Z_bias", &Z_bias); add_view("O_bias", &O_bias);
}


Lstm97Layer::FwdState::FwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    //Views on all activations
    Ia(NULL, n_cells, n_batches, time), Ib(NULL, n_cells, n_batches, time), //!< Input gate activation
    Fa(NULL, n_cells, n_batches, time), Fb(NULL, n_cells, n_batches, time), //!< forget gate activation
    Oa(NULL, n_cells, n_batches, time), Ob(NULL, n_cells, n_batches, time), //!< output gate activation

    Za(NULL, n_cells, n_batches, time), Zb(NULL, n_cells, n_batches, time), //!< Za =Net Activation, Zb=f(Za)
    S(NULL, n_cells, n_batches, time),      //!< Sa =Cell State activations
    f_S(NULL, n_cells, n_batches, time),      //!< Sa =Cell State activations
    Hb(NULL, n_cells, n_batches, time),     //!< output of LSTM block
    tmp1(NULL, n_cells, n_batches, time) // for calculating derivs
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


Lstm97Layer::BwdState::BwdState(size_t, size_t n_cells, size_t n_batches, size_t time) :
    //Views on all activations
    Ia(n_cells, n_batches, time), Ib(n_cells, n_batches, time), //Input gate activation
    Fa(n_cells, n_batches, time), Fb(n_cells, n_batches, time), //forget gate activation
    Oa(n_cells, n_batches, time), Ob(n_cells, n_batches, time), //output gate activation

    Za(n_cells, n_batches, time), Zb(n_cells, n_batches, time), //Net Activation
    S(n_cells, n_batches, time), //Cell activations
    f_S(n_cells, n_batches, time), //cell state activations
    Hb(n_cells, n_batches, time),     //!< output of LSTM block

    tmp1(n_cells, n_batches, time) // for calculating derivs
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


void Lstm97Layer::forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y, bool) {
    mult(w.ZX, x.slice(1,x.n_slices).flatten_time(), b.Za.slice(1,b.Za.n_slices).flatten_time());
    if (input_gate) {
        mult(w.IX, x.slice(1,x.n_slices).flatten_time(), b.Ia.slice(1,b.Ia.n_slices).flatten_time());
    }
    if (forget_gate) {
        mult(w.FX, x.slice(1,x.n_slices).flatten_time(), b.Fa.slice(1,b.Fa.n_slices).flatten_time());
    }
    if (output_gate) {
        mult(w.OX, x.slice(1,x.n_slices).flatten_time(), b.Oa.slice(1,b.Oa.n_slices).flatten_time());
    }
    for (size_t t(1); t < x.n_slices; ++t) {
        // RAW ACTIVATIONS
        mult_add(w.ZH, y.slice(t - 1), b.Za.slice(t));
        if (input_gate) {
            mult_add(w.IH, y.slice(t - 1), b.Ia.slice(t));
        }
        if (forget_gate) {
            mult_add(w.FH, y.slice(t - 1), b.Fa.slice(t));
        }
        if (output_gate) {
            mult_add(w.OH, y.slice(t - 1), b.Oa.slice(t));
        }

        if (use_bias) {
            add_vector_into(w.Z_bias, b.Za.slice(t));
            if (input_gate) {
                add_vector_into(w.I_bias, b.Ia.slice(t));
            }
            if (forget_gate) {
                add_vector_into(w.F_bias, b.Fa.slice(t));
            }
            if (output_gate) {
                add_vector_into(w.O_bias, b.Oa.slice(t));
            }
        }

        if (gate_recurrence) {
            if (input_gate) {
                mult_add(w.II, b.Ib.slice(t - 1), b.Ia.slice(t));
                if (forget_gate) {
                    mult_add(w.FI, b.Ib.slice(t - 1), b.Fa.slice(t));
                }
                if (output_gate) {
                    mult_add(w.OI, b.Ib.slice(t - 1), b.Oa.slice(t));
                }
            }
            if (forget_gate) {
                if (input_gate) {
                    mult_add(w.IF, b.Fb.slice(t - 1), b.Ia.slice(t));
                }
                mult_add(w.FF, b.Fb.slice(t - 1), b.Fa.slice(t));
                if (output_gate) {
                    mult_add(w.OF, b.Fb.slice(t - 1), b.Oa.slice(t));
                }
            }
            if (output_gate) {
                if (input_gate) {
                    mult_add(w.IO, b.Ob.slice(t - 1), b.Ia.slice(t));
                }
                if (forget_gate) {
                    mult_add(w.FO, b.Ob.slice(t - 1), b.Fa.slice(t));
                }
                mult_add(w.OO, b.Ob.slice(t - 1), b.Oa.slice(t));
            }
        }

        if (peephole_connections) {
            if (input_gate) {
                dot_add(b.S.slice(t - 1), w.IS, b.Ia.slice(t));
            }
            if (forget_gate) {
                dot_add(b.S.slice(t - 1), w.FS, b.Fa.slice(t));
            }
        }

        // ACTIVATION FUNCTIONS
        apply_tanh(b.Za.slice(t), b.Zb.slice(t));
        if (input_gate) {
            apply_sigmoid(b.Ia.slice(t), b.Ib.slice(t));
            dot(b.Zb.slice(t), b.Ib.slice(t), b.S.slice(t));
        } else {
            copy(b.Zb.slice(t), b.S.slice(t));
        }

        if (forget_gate) {
            apply_sigmoid(b.Fa.slice(t), b.Fb.slice(t));
            dot_add(b.S.slice(t - 1), b.Fb.slice(t), b.S.slice(t));
        } else {
	        add_into_b(b.S.slice(t - 1), b.S.slice(t));
	    }

        // cell
        f->apply(b.S.slice(t), b.f_S.slice(t));

        if (output_gate) {
            if (peephole_connections) {
                dot_add(b.S.slice(t), w.OS, b.Oa.slice(t));
            }
            apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
            dot(b.f_S.slice(t), b.Ob.slice(t), y.slice(t));
        }
        else {
            copy(b.f_S.slice(t), y.slice(t));
        }

    }
}


void Lstm97Layer::backward(Parameters& w, FwdState& b, BwdState& d, Matrix& y, Matrix& in_deltas, Matrix& out_deltas) {
    dampened_backward(w, b, d, y, in_deltas, out_deltas, b, 0., 0.);
}


void Lstm97Layer::gradient(Parameters&, Parameters& grad, FwdState& b, BwdState& d, Matrix& y, Matrix& x, Matrix& )  {
    size_t n_time = x.n_slices;

    mult(d.Za.slice(1,d.Za.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.ZX);
    if (input_gate) {
        mult(d.Ia.flatten_time(), x.flatten_time().T(), grad.IX);
    }
    if (forget_gate) {
        mult(d.Fa.slice(1,d.Fa.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.FX);
    }
    if (output_gate) {
        mult(d.Oa.slice(1,d.Oa.n_slices).flatten_time(), x.slice(1,x.n_slices).flatten_time().T(), grad.OX);
    }


    if (n_time > 1) {
        grad.ZH.set_all_elements_to(0.0);
        grad.IH.set_all_elements_to(0.0);
        grad.FH.set_all_elements_to(0.0);
        grad.OH.set_all_elements_to(0.0);
        for (int t = 0; t < n_time - 1; ++t) {
            mult_add(d.Za.slice(t + 1), y.slice(t).T(), grad.ZH);
            if (input_gate) {
                mult_add(d.Ia.slice(t + 1), y.slice(t).T(), grad.IH);
            }
            if (forget_gate) {
                mult_add(d.Fa.slice(t + 1), y.slice(t).T(), grad.FH);
            }
            if (output_gate) {
                mult_add(d.Oa.slice(t + 1), y.slice(t).T(), grad.OH);
            }
        }
	    if (gate_recurrence) {
	        if (input_gate) {
                mult(d.Ia.slice(1, n_time).flatten_time(), b.Ib.slice(0, n_time - 1).flatten_time().T(), grad.II);
                if (forget_gate) {
                    mult(d.Ia.slice(1, n_time).flatten_time(), b.Fb.slice(0, n_time - 1).flatten_time().T(), grad.IF);
                }
                if (output_gate) {
                    mult(d.Ia.slice(1, n_time).flatten_time(), b.Ob.slice(0, n_time - 1).flatten_time().T(), grad.IO);
                }
            }

            if (forget_gate) {
                if (input_gate) {
                    mult(d.Fa.slice(1, n_time).flatten_time(), b.Ib.slice(0, n_time - 1).flatten_time().T(), grad.FI);
                }
                mult(d.Fa.slice(1, n_time).flatten_time(), b.Fb.slice(0, n_time - 1).flatten_time().T(), grad.FF);
                if (output_gate) {
                    mult(d.Fa.slice(1, n_time).flatten_time(), b.Ob.slice(0, n_time - 1).flatten_time().T(), grad.FO);
                }
            }
            if (output_gate) {
                if (input_gate) {
                    mult(d.Oa.slice(1, n_time).flatten_time(), b.Ib.slice(0, n_time - 1).flatten_time().T(), grad.OI);
                }
                if (forget_gate) {
                    mult(d.Oa.slice(1, n_time).flatten_time(), b.Fb.slice(0, n_time - 1).flatten_time().T(), grad.OF);
                }
                mult(d.Oa.slice(1, n_time).flatten_time(), b.Ob.slice(0, n_time - 1).flatten_time().T(), grad.OO);
            }
        }
    }

    if (n_time > 1 && peephole_connections) {
        if (forget_gate) {
            dot_squash(d.Fa.slice(1, n_time), b.S.slice(0, n_time - 1), grad.FS);
        }
        if (input_gate) {
            dot_squash(d.Ia.slice(1, n_time), b.S.slice(0, n_time - 1), grad.IS);
        }
    }

    if (peephole_connections && output_gate) {
        dot_squash(d.Oa, b.S, grad.OS);
    }

    if (use_bias) {
        squash(d.Za.slice(1, d.Za.n_slices), grad.Z_bias);
        if (input_gate) {
            squash(d.Ia.slice(1, d.Za.n_slices), grad.I_bias);
        }
        if (forget_gate) {
            squash(d.Fa.slice(1, d.Za.n_slices), grad.F_bias);
        }
        if (output_gate) {
            squash(d.Oa.slice(1, d.Za.n_slices), grad.O_bias);
        }
    }
}


void Lstm97Layer::Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry) {

    mult(v.ZX, x.slice(1,x.n_slices).flatten_time(), Rb.Za.slice(1,Rb.Za.n_slices).flatten_time());
    mult_add(w.ZX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Za.slice(1,Rb.Za.n_slices).flatten_time());
    if (input_gate) {
        mult(v.IX, x.slice(1,x.n_slices).flatten_time(), Rb.Ia.slice(1,Rb.Ia.n_slices).flatten_time());
        mult_add(w.IX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Ia.slice(1,Rb.Ia.n_slices).flatten_time());
    }
    if (forget_gate) {
        mult(v.FX, x.slice(1,x.n_slices).flatten_time(), Rb.Fa.slice(1,Rb.Fa.n_slices).flatten_time());
        mult_add(w.FX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Fa.slice(1,Rb.Fa.n_slices).flatten_time());
    }
    if (output_gate) {
        mult(v.OX, x.slice(1,x.n_slices).flatten_time(), Rb.Oa.slice(1,Rb.Oa.n_slices).flatten_time());
        mult_add(w.OX, Rx.slice(1,Rx.n_slices).flatten_time(), Rb.Oa.slice(1,Rb.Oa.n_slices).flatten_time());
    }

    for (size_t t(1); t < x.n_slices; ++t) {
        mult_add(w.ZH, Ry.slice(t - 1), Rb.Za.slice(t));
        mult_add(v.ZH, y.slice(t - 1), Rb.Za.slice(t));

        if (input_gate) {
            mult_add(w.IH, Ry.slice(t - 1), Rb.Ia.slice(t));
            mult_add(v.IH, y.slice(t - 1), Rb.Ia.slice(t));
            if (gate_recurrence) {
                mult_add(w.II, Rb.Ib.slice(t - 1), Rb.Ia.slice(t));
                mult_add(v.II, b.Ib.slice(t - 1), Rb.Ia.slice(t));
                if (forget_gate) {
                    mult_add(w.IF, Rb.Fb.slice(t - 1), Rb.Ia.slice(t));
                    mult_add(v.IF, b.Fb.slice(t - 1), Rb.Ia.slice(t));
                }
                if (output_gate) {
                    mult_add(w.IO, Rb.Ob.slice(t - 1), Rb.Ia.slice(t));
                    mult_add(v.IO, b.Ob.slice(t - 1), Rb.Ia.slice(t));
                }
            }
	    }

        if (forget_gate) {
            mult_add(w.FH, Ry.slice(t - 1), Rb.Fa.slice(t));
            mult_add(v.FH, y.slice(t - 1), Rb.Fa.slice(t));
            if (gate_recurrence) {
                if (input_gate) {
                    mult_add(w.FI, Rb.Ib.slice(t - 1), Rb.Fa.slice(t));
                    mult_add(v.FI, b.Ib.slice(t - 1), Rb.Fa.slice(t));
                }
                mult_add(w.FF, Rb.Fb.slice(t - 1), Rb.Fa.slice(t));
                mult_add(v.FF, b.Fb.slice(t - 1), Rb.Fa.slice(t));

                if (output_gate) {
                    mult_add(w.FO, Rb.Ob.slice(t - 1), Rb.Fa.slice(t));
                    mult_add(v.FO, b.Ob.slice(t - 1), Rb.Fa.slice(t));
                }
            }
        }

        if (output_gate) {
            mult_add(w.OH, Ry.slice(t - 1), Rb.Oa.slice(t));
            mult_add(v.OH, y.slice(t - 1), Rb.Oa.slice(t));
            if (gate_recurrence) {
                if (input_gate) {
                    mult_add(w.OI, Rb.Ib.slice(t - 1), Rb.Oa.slice(t));
                    mult_add(v.OI, b.Ib.slice(t - 1), Rb.Oa.slice(t));
                }
                if (forget_gate) {
	                mult_add(w.OF, Rb.Fb.slice(t - 1), Rb.Oa.slice(t));
	                mult_add(v.OF, b.Fb.slice(t - 1), Rb.Oa.slice(t));
	            }
	            mult_add(w.OO, Rb.Ob.slice(t - 1), Rb.Oa.slice(t));
	            mult_add(v.OO, b.Ob.slice(t - 1), Rb.Oa.slice(t));
	        }
	    }

        if (peephole_connections) {
            if (input_gate) {
                dot_add(Rb.S.slice(t - 1), w.IS, Rb.Ia.slice(t));
                dot_add(b.S.slice(t - 1), v.IS, Rb.Ia.slice(t));
            }
            if (forget_gate) {
                dot_add(Rb.S.slice(t - 1), w.FS, Rb.Fa.slice(t));
                dot_add(b.S.slice(t - 1), v.FS, Rb.Fa.slice(t));
            }
        }

        if (use_bias) {
            if (input_gate) {
                add_vector_into(v.I_bias, Rb.Ia.slice(t));
            }
            if (forget_gate) {
                add_vector_into(v.F_bias, Rb.Fa.slice(t));
            }
            add_vector_into(v.Z_bias, Rb.Za.slice(t));
            if (output_gate) {
                add_vector_into(v.O_bias, Rb.Oa.slice(t));
            }
        }

        apply_tanh_deriv(b.Zb.slice(t), Rb.tmp1.slice(t));
        dot(Rb.tmp1.slice(t), Rb.Za.slice(t), Rb.Zb.slice(t));

        if (input_gate) {
            apply_sigmoid_deriv(b.Ib.slice(t), Rb.tmp1.slice(t));
            dot(Rb.tmp1.slice(t), Rb.Ia.slice(t), Rb.Ib.slice(t));

            dot(Rb.Ib.slice(t), b.Zb.slice(t), Rb.S.slice(t));
            dot_add(b.Ib.slice(t), Rb.Zb.slice(t), Rb.S.slice(t));
        } else {
            copy(b.Zb.slice(t), Rb.S.slice(t));
            add_into_b(Rb.Zb.slice(t), Rb.S.slice(t));
        }


        if (forget_gate) {
            apply_sigmoid_deriv(b.Fb.slice(t), Rb.tmp1.slice(t));
            dot(Rb.tmp1.slice(t), Rb.Fa.slice(t), Rb.Fb.slice(t));

            dot_add(Rb.S.slice(t - 1), b.Fb.slice(t), Rb.S.slice(t));
            dot_add(b.S.slice(t - 1), Rb.Fb.slice(t), Rb.S.slice(t));
        } else {
            add_into_b(Rb.S.slice(t - 1), Rb.S.slice(t));
        }

        if (output_gate) {
            if (peephole_connections) {
                dot_add(Rb.S.slice(t), w.OS, Rb.Oa.slice(t));
                dot_add(b.S.slice(t), v.OS, Rb.Oa.slice(t));
            }
            apply_sigmoid_deriv(b.Ob.slice(t), Rb.tmp1.slice(t));
            dot(Rb.tmp1.slice(t), Rb.Oa.slice(t), Rb.Ob.slice(t));

            f->apply_deriv(b.f_S.slice(t), Rb.S.slice(t), Rb.tmp1.slice(t));
            dot(Rb.tmp1.slice(t), b.Ob.slice(t), Ry.slice(t));
            dot_add(Rb.Ob.slice(t), b.f_S.slice(t), Ry.slice(t));
        } else {
            f->apply_deriv(b.f_S.slice(t), Rb.S.slice(t), Ry.slice(t));
        }
    }
}


//instead of normal deltas buffer, pass in empty Rdeltas buffer, and instead of out_deltas, pass in the Ry value calculated by the Rfwd pass
void Lstm97Layer::dampened_backward(Parameters &w, FwdState &b, BwdState &d, Matrix& y, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu) {
    int end_time = static_cast<int>(y.n_slices - 1);
    copy(out_deltas, d.Hb);

    for(int t(end_time); t >= 0; --t) {
        if (t<end_time) {
            if (full_gradient) {
                mult_add(w.ZH.T(), d.Za.slice(t+1), d.Hb.slice(t));
                if (input_gate) {
                    mult_add(w.IH.T(), d.Ia.slice(t+1), d.Hb.slice(t));
                }
                if (forget_gate) {
                    mult_add(w.FH.T(), d.Fa.slice(t+1), d.Hb.slice(t));
                }
                if (output_gate) {
                    mult_add(w.OH.T(), d.Oa.slice(t+1), d.Hb.slice(t));
                }
            }

            if (forget_gate) {
                dot(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
            } else {
                copy(d.S.slice(t+1), d.S.slice(t));
            }

            if (peephole_connections && full_gradient) {
                if (input_gate) {
                    dot_add(d.Ia.slice(t+1), w.IS, d.S.slice(t));
                }
                if (forget_gate) {
                    dot_add(d.Fa.slice(t+1), w.FS, d.S.slice(t));
                }
            }
        }

        if (output_gate) {
            dot(d.Hb.slice(t), b.Ob.slice(t), d.f_S.slice(t));
            dot(d.Hb.slice(t), b.f_S.slice(t), d.Ob.slice(t));

            if (full_gradient && gate_recurrence) {
                if (t < end_time) {
                    if (input_gate) {
                        mult_add(w.IO.T(), d.Ia.slice(t+1), d.Ob.slice(t));
                    }
                    if (forget_gate) {
                        mult_add(w.FO.T(), d.Fa.slice(t+1), d.Ob.slice(t));
                    }
                    mult_add(w.OO.T(), d.Oa.slice(t+1), d.Ob.slice(t));
                 }
            }

            apply_sigmoid_deriv(b.Ob.slice(t), d.tmp1.slice(t));
            dot(d.Ob.slice(t), d.tmp1.slice(t), d.Oa.slice(t));
        } else {
            copy(d.Hb.slice(t), d.f_S.slice(t));
        }

        f->apply_deriv(b.f_S.slice(t), d.f_S.slice(t), d.tmp1.slice(t));

        if(t<end_time) {
            add_into_b(d.tmp1.slice(t), d.S.slice(t));
        } else {
            copy(d.tmp1.slice(t), d.S.slice(t));
        }

        if (peephole_connections && output_gate) {
            dot_add(d.Oa.slice(t), w.OS, d.S.slice(t));
        }

        if (input_gate) {
            dot(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
        } else {
            copy(d.S.slice(t), d.Zb.slice(t));
        }
        apply(b.Zb.slice(t), d.tmp1.slice(t), &tanh_deriv);
        dot(d.Zb.slice(t), d.tmp1.slice(t), d.Za.slice(t));


        if (input_gate) {
            dot(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));
        }

        if (full_gradient && gate_recurrence && input_gate) {
            if (t<end_time) {
                mult_add(w.II.T(), d.Ia.slice(t+1), d.Ib.slice(t));
                if (forget_gate) {
                    mult_add(w.FI.T(), d.Fa.slice(t+1), d.Ib.slice(t));
                }
                if (output_gate) {
                    mult_add(w.OI.T(), d.Oa.slice(t+1), d.Ib.slice(t));
                }
            }
        }

        if (input_gate) {
            apply_sigmoid_deriv(b.Ib.slice(t), d.tmp1.slice(t));
            dot(d.Ib.slice(t), d.tmp1.slice(t), d.Ia.slice(t));
        }

        if (forget_gate) {
            if (t) {
                dot(d.S.slice(t), b.S.slice(t - 1), d.Fb.slice(t));
            } else {
                d.Fb.slice(t).set_all_elements_to(0.0);
            }
        }

        if (full_gradient && forget_gate && gate_recurrence && t < end_time) {
            mult_add(w.FF.T(), d.Fa.slice(t+1), d.Fb.slice(t));
            if (input_gate) {
                mult_add(w.IF.T(), d.Ia.slice(t+1), d.Fb.slice(t));
            }
            if (output_gate) {
                mult_add(w.OF.T(), d.Oa.slice(t+1), d.Fb.slice(t));
            }
        }

        if (forget_gate) {
            apply_sigmoid_deriv(b.Fb.slice(t), d.tmp1.slice(t));
            dot(d.Fb.slice(t), d.tmp1.slice(t), d.Fa.slice(t));
        }

        if(t) {
            mult_add(w.ZX.T(), d.Za.slice(t), in_deltas.slice(t));
            if (input_gate) {
                mult_add(w.IX.T(), d.Ia.slice(t), in_deltas.slice(t));
            }
            if (forget_gate) {
                mult_add(w.FX.T(), d.Fa.slice(t), in_deltas.slice(t));
            }
            if (output_gate) {
                mult_add(w.OX.T(), d.Oa.slice(t), in_deltas.slice(t));
            }
        }
    }
}
