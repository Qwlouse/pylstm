#include "lstm_layer.h"
#include "matrix/matrix_operation_cpu.h"

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

      dot_add(b.S.slice(t - 1), w.FS, b.Fa.slice(t));
      dot_add(b.S.slice(t - 1), w.IS, b.Ia.slice(t));
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

    dot_add(b.f_S.slice(t), w.OS, b.Oa.slice(t));
    apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
    dot(b.f_S.slice(t), b.Ob.slice(t), y.slice(t));
  }
  
}


void lstm_backward(LstmWeights &w, LstmBuffers &b, LstmDeltas &d, MatrixView3DCPU &y, MatrixView3DCPU &in_deltas) {

  //clear_temp();
  //size_t end_time(b.batch_time - 1);
  size_t end_time(b.time - 1);

  //!< calculate t+1 values except for end_time+1 
  for(int t(end_time); t >= 0; --t){
    if (t<end_time) {
      /*
	multiply_transpose(w.IH, d.Ia.slice(t+1), d.Hb.slice(t));
	multiply_transpose(w.FH, d.Fa.slice(t+1), d.Hb.slice(t));
	multiply_transpose(w.ZH, d.Za.slice(t+1), d.Hb.slice(t));
	multiply_transpose(w.OH, d.Oa.slice(t+1), d.Hb.slice(t));
      */

      //!< dE/dS += dE/dS(t+1) * b_F(t+1)
      dot_add(d.S.slice(t+1), b.Fb.slice(t+1), d.S.slice(t));
      //!< dE/dS += dE/da_I(t+1) * W_IS
      dot_add(d.Ia.slice(t+1), w.IS, d.S.slice(t));
      //!< dE/dS += dE/da_F(t+1) * W_FS
      dot_add(d.Fa.slice(t+1), w.FS, d.S.slice(t));
    }

    //!< dE/df_S += dE/dH * b_O THIS IS WEIRD, IT GOES WITH NEXT LINE ??!?!
    dot_add(d.Hb.slice(t), b.Ob.slice(t), d.f_S.slice(t));
    
    //OUTPUT GATES DERIVS
    //!< dE/db_O = dE/dH * f(s) * f(a_O)
    dot_add(d.Hb.slice(t), d.f_S.slice(t), d.Ob.slice(t));
    //!< dE/da_O = dE/db_O * f'(a_O)
    //sigmoid_deriv(d.Ob.slice(t), b.Ob.slice(t), d.temp_hidden, d.temp_hidden2, d.Oa.slice(t)); //o = -o^2
      
    //!< dE/dS += dE/df_S * f'(s)  
    //tanh2_deriv(d.f_S.slice(t), b.S.slice(t), d.temp_hidden, d.S.slice(t));

    
    //!< dE/dS += dE/da_O * W_OS
    dot_add(d.Oa.slice(t), w.OS, d.S.slice(t));
    
    //!< CELL ACTIVATION DERIVS
    //!< dE/db_Z = dE/dS * b_I
    dot_add(d.S.slice(t), b.Ib.slice(t), d.Zb.slice(t));
    //!< dE/da_Z = dE/db_Z * f'(a_Z)
    //tanh2_deriv(d.Zb.slice(t), b.Zb.slice(t), d.temp_hidden, d.Za.slice(t));
    
    //!< dE/db_I = dE/dS * b_Z 
    dot_add(d.S.slice(t), b.Zb.slice(t), d.Ib.slice(t));
    //!< dE/da_I = dE/db_I * f'(a_I)
    //sigmoid_deriv(d.Ib.slice(t), b.Ib.slice(t), d.temp_hidden, d.temp_hidden2, d.Ia.slice(t));
      
    //forget deltas
    //IF NEXT
    if (t)
      //!< dE/db_F += dE/dS * s(t-1)
      dot_add(d.S.slice(t), b.S.slice(t - 1), d.Fb.slice(t));
    
    //dE/da_F = dE/db_F * f'(a_F)
    //sigmoid_deriv(d.Fb.slice(t), b.Fb.slice(t), d.temp_hidden, d.temp_hidden2, d.Fa.slice(t));    
    //back through time stuff
    //IF NEXT
    
   
  }
  //		std::cout << "DONE" << std::endl;
}




