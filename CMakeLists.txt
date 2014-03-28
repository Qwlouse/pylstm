###############################################################################
###                          PY_LSTM BUILD SCRIPT                           ###
###############################################################################
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#==============================================================================
# Project Settings
#==============================================================================
PROJECT(PY_LSTM)
SET(VERSION_MAJOR 0)
SET(VERSION_MINOR 1)


#==============================================================================
# Set OS Variable
#==============================================================================
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
   # Mac OS X specific code
   SET(OperatingSystem "Mac OS X")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # Linux specific code
    SET(OperatingSystem "Linux")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux") 

#==============================================================================
# Path Settings
#==============================================================================
# Read PY_LSTM_PATH from options, cache, environment variable, or . 
# (in that order)
IF(NOT PY_LSTM_PATH)
    SET(ENV_PY_LSTM_PATH $ENV{PY_LSTM_PATH})
    IF(ENV_PY_LSTM_PATH)
        FILE(TO_CMAKE_PATH $ENV{PY_LSTM_PATH} PY_LSTM_PATH)
    ELSE(ENV_PY_LSTM_PATH)
        SET(PY_LSTM_PATH ${CMAKE_HOME_DIRECTORY})
    ENDIF(ENV_PY_LSTM_PATH)
ENDIF(NOT PY_LSTM_PATH)

# Set Cmake module path
SET(CMAKE_MODULE_PATH "${PY_LSTM_PATH}/cmake")

#==============================================================================
# Options settings (including optional packages)
#==============================================================================

OPTION(USE_MATLAB_BLAS "Use Matlab BLAS Libraries (requires Matlab)" OFF)
IF (USE_MATLAB_BLAS)
    FIND_PACKAGE(Matlab2 REQUIRED)
    SET(BLAS_LIBRARIES ${MATLAB_LIBRARIES})
    INCLUDE_DIRECTORIES(${MATLAB_EXTERN_INCLUDE_PATH})
ENDIF (USE_MATLAB_BLAS)

OPTION(USE_OPENBLAS "Use OpenBlas if possible" ON)


IF (NOT USE_MATLAB_BLAS)
    IF (USE_OPENBLAS)
        FIND_PACKAGE(OpenBlas)
    ENDIF (USE_OPENBLAS)

    IF (OPENBLAS_FOUND)
        SET(BLAS_LIBRARIES ${OPENBLAS})
    ELSE (OPENBLAS_FOUND)
        ENABLE_LANGUAGE(Fortran)
        FIND_PACKAGE(BLAS REQUIRED)
    ENDIF (OPENBLAS_FOUND)
ENDIF (NOT USE_MATLAB_BLAS)


# Enable Asserts
OPTION(ASSERTS_ENABLED "Enable asserts" ON)

# Compile the tests
OPTION(UNIT_TESTS "Compile the unit tests (fetches GTest)" ON)
IF (UNIT_TESTS)
    # INCLUDE(GetGTest)
ENDIF(UNIT_TESTS)

# Build the Documentation using Doxygen
OPTION(DOCUMENTATION "Build the Documentation using Doxygen" OFF)
IF (DOCUMENTATION)
    INCLUDE(UseLATEX)
    FIND_PACKAGE(Doxygen)
ENDIF(DOCUMENTATION)

#==============================================================================
# Compiler Flags
#==============================================================================
SET( ENV{EXTERNAL_WARNINGS_NOT_ERRORS} TRUE)

IF(DEFINED MSVC)
    ADD_DEFINITIONS(/W3 /wd4996)
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    ADD_DEFINITIONS(-Wall -Wextra -Wno-sign-compare -ansi -fPIC -std=c++0x -g -ggdb#-Weffc++ -pedantic
                    -Wconversion )#-Werror)
ENDIF(DEFINED MSVC)

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    IF ("${OperatingSystem}" STREQUAL "Mac OS X")
   		# by Juxi for Mac (10.8)
		set( CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++" )
    	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++" )
	ELSE ("${OperatingSystem}" STREQUAL "Mac OS X")
		ADD_DEFINITIONS(-I/usr/lib/c++/v1/ -libstd=libc++ -std=c++11)
	ENDIF ("${OperatingSystem}" STREQUAL "Mac OS X")
	
    MESSAGE("--> We are using CLANG - Flags were set accordingly!")
ENDIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

#==============================================================================
# Config header
#==============================================================================
# configure a header file to pass some of the CMake settings to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/src/Config.h.in"
  "${PROJECT_BINARY_DIR}/src/Config.h"
  )
# add the to the search path for include files so that we will find Config.h
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}/src")

#==============================================================================
# Cython
#==============================================================================
# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
INCLUDE(UseCython)
INCLUDE_DIRECTORIES(/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/)
# We reproduce the availability of the source files in the build tree.
# That way we can run the unchanged setup.py from the build dir and install
# both cython and pure python files.
ADD_CUSTOM_TARGET( ReplicatePythonSourceTree ALL ${CMAKE_CURRENT_SOURCE_DIR}/replicate_python_sources.sh
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

#==============================================================================
# Process sources
#==============================================================================
ADD_SUBDIRECTORY(src)

#==============================================================================
# Testing
#==============================================================================
IF (UNIT_TESTS)
    ADD_SUBDIRECTORY (gtest-1.7.0)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
    ADD_CUSTOM_TARGET(pytest
                      nosetests
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pylstm
                      COMMENT "Running all python tests using nosetest"
                      )
ENDIF(UNIT_TESTS)

#==============================================================================
# Documentation
#==============================================================================
IF (DOCUMENTATION)
    ADD_SUBDIRECTORY(doc EXCLUDE_FROM_ALL)
    # configure doxygen configuration file
    CONFIGURE_FILE(
        "${PROJECT_SOURCE_DIR}/config/doxygen.conf.in"
        "${PROJECT_BINARY_DIR}/config/doxygen.conf"
        @ONLY
    )

    ADD_CUSTOM_TARGET(doc
                      ${DOXYGEN_EXECUTABLE} 
                      ${PROJECT_BINARY_DIR}/config/doxygen.conf
                      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc
                      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM
                      )
    ADD_DEPENDENCIES(doc pdf)
ENDIF(DOCUMENTATION)

