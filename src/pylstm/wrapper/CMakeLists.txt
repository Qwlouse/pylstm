SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/pylstm/wrapper)
# If the pyx file is a C++ file, we should specify that here.
SET_SOURCE_FILES_PROPERTIES(
  py_matrix.pyx
  py_layers.pyx
  py_view_container.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/matrix)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/layers)

CYTHON_ADD_MODULE(py_matrix py_matrix.pyx)
TARGET_LINK_LIBRARIES(py_matrix matrix)

CYTHON_ADD_MODULE(py_view_container py_view_container.pyx)
TARGET_LINK_LIBRARIES(py_view_container layers)

CYTHON_ADD_MODULE(py_layers py_layers.pyx)
TARGET_LINK_LIBRARIES(py_layers layers)
