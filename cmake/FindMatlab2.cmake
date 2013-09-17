#  Find module for MATLAB
###############################################################################
# This module defines:
# - MATLAB_EXTERN_INCLUDE_PATH
# - MATLAB_LIBRARIES
# - MATLAB_FOUND
#
###############################################################################

# List of variables, which will set:
set(_SEARCHED_VARIABLES
    MATLAB_EXTERN_INCLUDE_PATH
    )

# Force quiet, if everything is found:
set(_QUIET_FINDING true)
foreach(VAR ${_SEARCHED_VARIABLES})
    if(NOT ${VAR})
        set(_QUIET_FINDING false)
    endif(NOT ${VAR})
endforeach(VAR)

if(_QUIET_FINDING)
    set(MATLAB_FIND_QUIETLY ON)
endif(_QUIET_FINDING)


#------------------------------------------------------------------------------
# MATLAB_INCLUDE_PATH
#------------------------------------------------------------------------------

IF(OperatingSystem MATCHES "Mac OS X")
    find_path(MATLAB_EXTERN_INCLUDE_PATH
    NAMES blas.h mex.h mat.h
    PATH_SUFFIXES MATLAB_R2011a.app/extern/include/ MATLAB_R2011b.app/extern/include/ MATLAB_R2012a.app/extern/include/ MATLAB_R2012b.app/extern/include/ MATLAB_R2013a.app/extern/include/
    PATHS /Applications 
    )
ENDIF(OperatingSystem MATCHES "Mac OS X")

IF(OperatingSystem MATCHES "Linux")
find_path(MATLAB_EXTERN_INCLUDE_PATH
    NAMES blas.h mex.h mat.h
    PATH_SUFFIXES R2012b/extern/include R2012a/extern/include R2011a/extern/include R2011b/extern/include 
    PATHS /usr/local/MATLAB ~/libs/MATLAB
    )
ENDIF(OperatingSystem MATCHES "Linux")

#------------------------------------------------------------------------------
# MATLAB_LIBRARIES
#------------------------------------------------------------------------------

IF(OperatingSystem MATCHES "Mac OS X")
FIND_LIBRARY(MATLAB_LIBRARIES
    NAMES mwblas
    PATH_SUFFIXES MATLAB_R2011a.app/bin/maci64 MATLAB_R2011b.app/bin/maci64 MATLAB_R2012a.app/bin/maci64 MATLAB_R2012b.app/bin/maci64 MATLAB_R2013a.app/bin/maci64
    PATHS /Applications 
    )

FIND_LIBRARY(MATLAB_LIBRARIES
    NAMES iomp5
    PATH_SUFFIXES MATLAB_R2011a.app/bin/maci64 MATLAB_R2011b.app/bin/maci64 MATLAB_R2012a.app/bin/maci64 MATLAB_R2012b.app/bin/maci64
    PATHS /Applications 
)

ENDIF(OperatingSystem MATCHES "Mac OS X")

IF(OperatingSystem MATCHES "Linux")
FIND_LIBRARY(MATLAB_LIBRARIES
    NAMES mwblas
    PATH_SUFFIXES R2012b/bin/glnxa64 R2012a/bin/glnxa64 R2011b/bin/glnxa64 R2011a/bin/glnxa64
    PATHS /usr/local/MATLAB ~/libs/MATLAB
    )
ENDIF(OperatingSystem MATCHES "Linux")

#------------------------------------------------------------------------------
# MATLAB_FOUND
#------------------------------------------------------------------------------

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MATLAB DEFAULT_MSG ${_SEARCHED_VARIABLES})
MARK_AS_ADVANCED(${_SEARCHED_VARIABLES})
