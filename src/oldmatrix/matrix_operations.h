/*
 * matrix_operations.h
 *
 *  Created on: Jun 14, 2011
 *      Author: stollenga
 */

#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#include "defines.h"

#ifdef USE_CBLAS
//cblas operations
#include "matrix_operations_cblas.h"
#endif

#ifdef USE_MATLABBLAS
//matlab blas operations
#include "matrix_operations_matlabblas.h"
#endif

#ifdef USE_GPU
//cuda operations
#include "matrix_operations_cuda.h"
#include "apply_function_cuda.cuh"
#endif

inline void sigmoid_deriv(MatrixPtr in_deltas, MatrixPtr activation, MatrixPtr temp_deriv, MatrixPtr temp_store, MatrixPtr deltas)
{
  //temp_deriv.clear();
  //temp_store.clear();
  copy(activation, temp_deriv); //o = -o
  scale(temp_deriv, -1.0);
  add(temp_deriv, 1.0); //o = (1.0 - o)
  multiply_vector(activation, temp_deriv, temp_store, 0.0); // o = (1.0 - o) * o
  multiply_vector(in_deltas, temp_store, deltas, 1.0);
}

inline void tanh_deriv(MatrixPtr in_deltas, MatrixPtr activations, MatrixPtr temp_deriv, MatrixPtr deltas)
{
  //temp_deriv.clear();
  square(activations, temp_deriv, -1.0); //o = -o^2
  add(temp_deriv, 1.0); //o = (1.0 - o^2)    
  multiply_vector(in_deltas, temp_deriv, deltas, 1.0);
}

inline void tanh2_deriv(MatrixPtr in_deltas, MatrixPtr activations, MatrixPtr temp_deriv, MatrixPtr deltas)
{
  //temp_deriv.clear();
  square(activations, temp_deriv, -.5); //o = -o^2
  add(temp_deriv, 2.0); //o = (1.0 - o^2)    
  multiply_vector(in_deltas, temp_deriv, deltas, 1.0);
}


#ifdef USE_GPU
inline void sigmoid_deriv(MatrixPtrGPU in_deltas, MatrixPtrGPU activation, MatrixPtrGPU temp_deriv, MatrixPtrGPU temp_store, MatrixPtrGPU deltas)
{
  //temp_deriv.clear();
  //temp_store.clear();
  copy(activation, temp_deriv); //o = -o
  scale(temp_deriv, -1.0);
  add(temp_deriv, 1.0); //o = (1.0 - o)
  multiply_vector(activation, temp_deriv, temp_store, 0.0); // o = (1.0 - o) * o
  multiply_vector(in_deltas, temp_store, deltas, 1.0);
}

inline void tanh_deriv(MatrixPtrGPU in_deltas, MatrixPtrGPU activations, MatrixPtrGPU temp_deriv, MatrixPtrGPU deltas)
{
  //temp_deriv.clear();
  square(activations, temp_deriv, -1.0); //o = -o^2
  add(temp_deriv, 1.0); //o = (1.0 - o^2)
  multiply_vector(in_deltas, temp_deriv, deltas, 1.0);
}


inline void tanh2_deriv(MatrixPtrGPU in_deltas, MatrixPtrGPU activations, MatrixPtrGPU temp_deriv, MatrixPtrGPU deltas)
{
  //temp_deriv.clear();
  square(activations, temp_deriv, -.5); //o = -o^2
  add(temp_deriv, 2.0); //o = (1.0 - o^2)
  multiply_vector(in_deltas, temp_deriv, deltas, 1.0);
}
#endif



#endif /* MAMatrixPtrRIX_OPERAMatrixPtrIONS_H_ */
