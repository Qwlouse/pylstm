#include "lstm_layer.h"

#include <iostream>
#include <vector>

#include "Core.h"
#include "matrix/matrix_operation.h"


LstmLayer::LstmLayer():
	f(&Tanhx2)
{ }

LstmLayer::LstmLayer(const ActivationFunction* f):
	f(f)
{ }

LstmLayer::Parameters::Parameters(size_t n_inputs_, size_t n_cells_) :
    n_inputs(n_inputs_),
    n_cells(n_cells_),
    IX(NULL, n_cells, n_inputs, 1), IH(NULL, n_cells, n_cells, 1), IS(NULL, 1, n_cells, 1),
    FX(NULL, n_cells, n_inputs, 1), FH(NULL, n_cells, n_cells, 1), FS(NULL, 1, n_cells, 1),
    ZX(NULL, n_cells, n_inputs, 1), ZH(NULL, n_cells, n_cells, 1),
    OX(NULL, n_cells, n_inputs, 1), OH(NULL, n_cells, n_cells, 1), OS(NULL, 1, n_cells, 1),
    I_bias(NULL, n_cells, 1, 1), F_bias(NULL, n_cells, 1, 1), Z_bias(NULL, n_cells, 1, 1), O_bias(NULL, n_cells, 1, 1)
{

    add_view("IX", &IX); add_view("IH", &IH); add_view("IS", &IS);
    add_view("FX", &FX); add_view("FH", &FH); add_view("FS", &FS);
    add_view("ZX", &ZX); add_view("ZH", &ZH);
    add_view("OX", &OX); add_view("OH", &OH); add_view("OS", &OS);
    add_view("I_bias", &I_bias); add_view("F_bias", &F_bias); add_view("Z_bias", &Z_bias); add_view("O_bias", &O_bias);
}

LstmLayer::FwdState::FwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
    n_inputs(n_inputs_), n_cells(n_cells_),
    n_batches(n_batches_), time(time_),

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

LstmLayer::BwdState::BwdState(size_t n_inputs_, size_t n_cells_, size_t n_batches_, size_t time_) :
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


void LstmLayer::forward(Parameters &w, FwdState &b, Matrix &x, Matrix &y) {
	size_t n_inputs = w.n_inputs;
	size_t n_cells = w.n_cells;
	size_t n_batches = b.n_batches;
	size_t n_slices = b.time;
	ASSERT(b.n_cells == n_cells);
	ASSERT(b.n_inputs == n_inputs);

	ASSERT(x.n_rows == n_inputs);
	ASSERT(x.n_columns == n_batches);
	ASSERT(x.n_slices == n_slices);

	ASSERT(y.n_rows == n_cells);
	ASSERT(y.n_columns == n_batches);
	ASSERT(y.n_slices == n_slices);


    mult(w.IX, x.flatten_time(), b.Ia.flatten_time());
    mult(w.FX, x.flatten_time(), b.Fa.flatten_time());
    mult(w.ZX, x.flatten_time(), b.Za.flatten_time());
    mult(w.OX, x.flatten_time(), b.Oa.flatten_time());

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
        apply_tanh(b.Za.slice(t), b.Zb.slice(t));
        dot(b.Zb.slice(t), b.Ib.slice(t), b.S.slice(t));

        if (t)
            dot_add(b.S.slice(t - 1), b.Fb.slice(t), b.S.slice(t));
        f->apply(b.S.slice(t), b.f_S.slice(t));
        dot_add(b.S.slice(t), w.OS, b.Oa.slice(t));

        //mult_add(b.S.slice(t), w.OS, b.Oa.slice(t));
        apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
        //copy(b.Oa.slice(t), b.Ob.slice(t));

        dot(b.f_S.slice(t), b.Ob.slice(t), y.slice(t));
    }
}


void LstmLayer::backward(Parameters& w, FwdState& b, BwdState& d, Matrix&, Matrix& in_deltas, Matrix& out_deltas) {

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
            //dot_add(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
            dot(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));

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
        f->apply_deriv(b.f_S.slice(t), d.f_S.slice(t), d.tmp1.slice(t));
        //dot_add(d.f_S.slice(t), d.tmp1.slice(t), d.S.slice(t));

        if(t<end_time)
            {add_into_b(d.tmp1.slice(t), d.S.slice(t));}
        else
            {copy(d.tmp1.slice(t), d.S.slice(t));}

        //! \f$\frac{dE}{dS} += \frac{dE}{da_O} * W_OS\f$
        dot_add(d.Oa.slice(t), w.OS, d.S.slice(t));

        //! CELL ACTIVATION DERIVS
        //! \f$\frac{dE}{db_Z} = \frac{dE}{dS} * b_I\f$
        dot(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
        //! \f$dE/da_Z = dE/db_Z * f'(a_Z)\f$
        apply(b.Zb.slice(t), d.tmp1.slice(t), &tanh_deriv);
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
        mult_add(w.IX.T(), d.Ia.slice(t), in_deltas.slice(t));
        mult_add(w.OX.T(), d.Oa.slice(t), in_deltas.slice(t));
        mult_add(w.ZX.T(), d.Za.slice(t), in_deltas.slice(t));
        mult_add(w.FX.T(), d.Fa.slice(t), in_deltas.slice(t));
    }
}

void LstmLayer::gradient(Parameters&, Parameters& grad, FwdState& b, BwdState& d, Matrix& y, Matrix& x, Matrix& )  {

    size_t n_time(b.time);

    //mult(d.output_deltas, d.Cb, delta_OT, 1.0 / n_time);

    //! \f$\frac{dE}{dW_ZX} += \frac{dE}{da_Z} * x(t)\f$
    //! \f$\frac{dE}{dW_FX} += \frac{dE}{da_F} * x(t)\f$
    //! \f$\frac{dE}{dW_IX} += \frac{dE}{da_I} * x(t)\f$
    //! \f$\frac{dE}{dW_OX} += \frac{dE}{da_O} * x(t)\f$
    mult(d.Za.flatten_time(), x.flatten_time().T(), grad.ZX); //  1.0 / 1.0); //(double) n_time);
    mult(d.Fa.flatten_time(), x.flatten_time().T(), grad.FX); // 1.0 / 1.0); //(double) n_time);
    mult(d.Ia.flatten_time(), x.flatten_time().T(), grad.IX); //1.0 / 1.0); //(double) n_time);
    mult(d.Oa.flatten_time(), x.flatten_time().T(), grad.OX); // 1.0 / 1.0); //(double) n_time);

    //! \f$\frac{dE}{dW_ZH} += \frac{dE}{da_Z} * h(t-1)\f$
    //! \f$\frac{dE}{dW_FH} += \frac{dE}{da_F} * h(t-1)\f$
    //! \f$\frac{dE}{dW_IH} += \frac{dE}{da_I} * h(t-1)\f$
    //! \f$\frac{dE}{dW_OH} += \frac{dE}{da_O} * h(t-1)\f$
    if (n_time > 1) {
        mult(d.Ia.slice(1, n_time-1).flatten_time(), y.slice(0, n_time - 2).flatten_time().T(), grad.IH); //(double) n_time);
        mult(d.Za.slice(1, n_time-1).flatten_time(), y.slice(0, n_time - 2).flatten_time().T(), grad.ZH); //(double) n_time);
        mult(d.Fa.slice(1, n_time-1).flatten_time(), y.slice(0, n_time - 2).flatten_time().T(), grad.FH); //(double) n_time);
        mult(d.Oa.slice(1, n_time-1).flatten_time(), y.slice(0, n_time - 2).flatten_time().T(), grad.OH); //(double) n_time);
    }


    //! \f$\frac{dE}{dW_FS} += \frac{dE}{da_F} * s(t-1)\f$
    //! \f$\frac{dE}{dW_IS} += \frac{dE}{da_I} * s(t-1)\f$
    if (n_time > 1) {
        dot_squash(d.Fa.slice(1, n_time-1), b.S.slice(0, n_time - 2), grad.FS);
        dot_squash(d.Ia.slice(1, n_time-1), b.S.slice(0, n_time - 2), grad.IS);
    }

    //! \f$\frac{dE}{dW_OS} += \frac{dE}{da_O} * s(t)\f$
    dot_squash(d.Oa, b.S, grad.OS);

    squash(d.Ia, grad.I_bias); //, 1.0 / (double) n_time);
    squash(d.Fa, grad.F_bias); //, 1.0 / (double) n_time);
    squash(d.Za, grad.Z_bias); //, 1.0 / (double) n_time);
    squash(d.Oa, grad.O_bias); //, 1.0 / (double)n_time); //, 1.0 / (double) n_time);

}

void LstmLayer::Rpass(Parameters &w, Parameters &v,  FwdState &b, FwdState &Rb, Matrix &x, Matrix &y, Matrix& Rx, Matrix &Ry) {

  mult(v.IX, x.flatten_time(), Rb.Ia.flatten_time());
  mult(v.FX, x.flatten_time(), Rb.Fa.flatten_time());
  mult(v.ZX, x.flatten_time(), Rb.Za.flatten_time());
  mult(v.OX, x.flatten_time(), Rb.Oa.flatten_time());

  mult_add(w.IX, Rx.flatten_time(), Rb.Ia.flatten_time());
  mult_add(w.FX, Rx.flatten_time(), Rb.Fa.flatten_time());
  mult_add(w.ZX, Rx.flatten_time(), Rb.Za.flatten_time());
  mult_add(w.OX, Rx.flatten_time(), Rb.Oa.flatten_time());


  for (size_t t(0); t < b.time; ++t) {
    
    //IF NEXT                                                                                 
    if (t) {
      mult_add(w.IH, Ry.slice(t - 1), Rb.Ia.slice(t));
      mult_add(w.FH, Ry.slice(t - 1), Rb.Fa.slice(t));
      mult_add(w.ZH, Ry.slice(t - 1), Rb.Za.slice(t));
      mult_add(w.OH, Ry.slice(t - 1), Rb.Oa.slice(t));

      mult_add(v.IH, y.slice(t - 1), Rb.Ia.slice(t));
      mult_add(v.FH, y.slice(t - 1), Rb.Fa.slice(t));
      mult_add(v.ZH, y.slice(t - 1), Rb.Za.slice(t));
      mult_add(v.OH, y.slice(t - 1), Rb.Oa.slice(t));

      dot_add(Rb.S.slice(t - 1), w.IS, Rb.Ia.slice(t));
      dot_add(Rb.S.slice(t - 1), w.FS, Rb.Fa.slice(t));

      dot_add(b.S.slice(t - 1), v.IS, Rb.Ia.slice(t));
      dot_add(b.S.slice(t - 1), v.FS, Rb.Fa.slice(t));

    }

    add_vector_into(v.I_bias, Rb.Ia.slice(t));
    add_vector_into(v.F_bias, Rb.Fa.slice(t));
    add_vector_into(v.Z_bias, Rb.Za.slice(t));
    add_vector_into(v.O_bias, Rb.Oa.slice(t));

    apply_sigmoid_deriv(b.Ib.slice(t), Rb.tmp1.slice(t));
    dot(Rb.tmp1.slice(t), Rb.Ia.slice(t), Rb.Ib.slice(t));

    apply_sigmoid_deriv(b.Fb.slice(t), Rb.tmp1.slice(t));
    dot(Rb.tmp1.slice(t), Rb.Fa.slice(t), Rb.Fb.slice(t));

    apply_tanh_deriv(b.Zb.slice(t), Rb.tmp1.slice(t));
    dot(Rb.tmp1.slice(t), Rb.Za.slice(t), Rb.Zb.slice(t));


    dot(Rb.Ib.slice(t), b.Zb.slice(t), Rb.S.slice(t));
    dot_add(b.Ib.slice(t), Rb.Zb.slice(t), Rb.S.slice(t));

    if (t) {
      dot_add(Rb.S.slice(t - 1), b.Fb.slice(t), Rb.S.slice(t));
      dot_add(b.S.slice(t - 1), Rb.Fb.slice(t), Rb.S.slice(t));
    }
  

    dot_add(Rb.S.slice(t), w.OS, Rb.Oa.slice(t));
    dot_add(b.S.slice(t), v.OS, Rb.Oa.slice(t));

    apply_sigmoid_deriv(b.Ob.slice(t), Rb.tmp1.slice(t));
    dot(Rb.tmp1.slice(t), Rb.Oa.slice(t), Rb.Ob.slice(t));


    f->apply_deriv(b.f_S.slice(t), Rb.S.slice(t), Rb.tmp1.slice(t));
    dot(Rb.tmp1.slice(t), b.Ob.slice(t), Ry.slice(t));
    dot_add(Rb.Ob.slice(t), b.f_S.slice(t), Ry.slice(t));
  }
}


//instead of normal deltas buffer, pass in empty Rdeltas buffer, and instead of out_deltas, pass in the Ry value calculated by the Rfwd pass
void LstmLayer::Rbackward(Parameters &w, FwdState &b, BwdState &d, Matrix &in_deltas, Matrix &out_deltas, FwdState &Rb, double lambda, double mu) {

  int end_time = static_cast<int>(b.time - 1);
  mu = 0;
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

