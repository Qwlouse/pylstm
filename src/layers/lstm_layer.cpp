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

      dot_add(b.Sb.slice(t - 1), w.FS, b.Fa.slice(t));
      dot_add(b.Sb.slice(t - 1), w.IS, b.Ia.slice(t));
    }

    add_into_b(w.F_bias, b.Fa.slice(t));
    add_into_b(w.I_bias, b.Ia.slice(t));
    add_into_b(w.Z_bias, b.Za.slice(t));
    add_into_b(w.O_bias, b.Oa.slice(t));

    apply_sigmoid(b.Fa.slice(t), b.Fb.slice(t));
    apply_sigmoid(b.Ia.slice(t), b.Ib.slice(t));
    apply_sigmoid(b.Za.slice(t), b.Zb.slice(t));
   
    dot_add(b.Zb.slice(t), b.Ib.slice(t), b.Sa.slice(t));
    if (t) 
      dot_add(b.Sa.slice(t - 1), b.Fb.slice(t), b.Sa.slice(t));
    apply_tanhx2(b.Sa.slice(t), b.Sb.slice(t));

    dot_add(b.Sb.slice(t), w.OS, b.Oa.slice(t));
    apply_sigmoid(b.Oa.slice(t), b.Ob.slice(t));
    dot(b.Sb.slice(t), b.Ob.slice(t), y.slice(t));
  }
}

