SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/pylstm)
# If the pyx file is a C++ file, we should specify that here.
SET_SOURCE_FILES_PROPERTIES(
  pylstm_wrapper.pyx
  PROPERTIES CYTHON_IS_CXX TRUE )

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/matrix)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/layers)

CYTHON_ADD_MODULE(pylstm_wrapper pylstm_wrapper.pyx)
TARGET_LINK_LIBRARIES(pylstm_wrapper matrix layers)


