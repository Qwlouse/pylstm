#include <string>
#include <iostream>
#include <gtest/gtest.h>

#include "Config.h"

#include "matrix/matrix_cpu.h"
#include "matrix/matrix_operation_cpu.h"
#include "layers/lstm_layer.h"

using namespace std;

TEST(MatrixLayer, lstm_forward) {
  size_t n_inputs(3), n_cells(5);
  size_t n_batches(1), n_time(10);

  LstmWeights w(n_inputs, n_cells);
  LstmBuffers b(n_inputs, n_cells, n_batches, n_time);
  LstmDeltas d(n_inputs, n_cells, n_batches, n_time);  
  
  MatrixCPU store_w(w.buffer_size(), 1, 1);
  MatrixCPU store_b(b.buffer_size(), 1, 1);
  MatrixCPU store_d(d.buffer_size(), 1, 1);
  
  cout << "allocating w" << endl;
  w.allocate(store_w);
  cout << "allocating b" << endl;
  b.allocate(store_b);
  cout << "allocating d" << endl;
  d.allocate(store_d);
  
  MatrixCPU x(n_inputs, n_batches, n_time);
  MatrixCPU y(n_cells, n_batches, n_time);
  MatrixCPU target(n_cells, n_batches, n_time);
  
  MatrixCPU in_deltas(n_inputs, n_batches, n_time);
  MatrixCPU out_deltas(n_cells, n_batches, n_time);


  cout << "running forward" << endl;
  lstm_forward(w, b, x, y);
  cout << "running backward" << endl;
  lstm_backward(w, b, d, y, in_deltas, out_deltas);
  cout << "done with tests" << endl;
  EXPECT_TRUE(true);
}
