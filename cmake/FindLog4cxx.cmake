# Find module for log4cxx
###############################################################################
# This module defines:
# - LOG4CXX_INCLUDE_PATH
# - LOG4CXX_LIBRARIES
# - LOG4CXX_FOUND
#
###############################################################################

# List of variables, which will set:
set(_SEARCHED_VARIABLES
    LOG4CXX_INCLUDE_PATH
    LOG4CXX_LIBRARIES
    )

# Force quiet, if everything is found:
set(_QUIET_FINDING true)
foreach(VAR ${_SEARCHED_VARIABLES})
    if(NOT ${VAR})
        set(_QUIET_FINDING false)
    endif(NOT ${VAR})
endforeach(VAR)

if(_QUIET_FINDING)
    set(LOG4CXX_FIND_QUIETLY ON)
endif(_QUIET_FINDING)


#------------------------------------------------------------------------------
# LOG4CXX_INCLUDE_PATH
#------------------------------------------------------------------------------

find_path(LOG4CXX_INCLUDE_PATH
    NAMES log4cxx/logger.h
    DOC "Path to the location of the log4cxx headers."
    )

#------------------------------------------------------------------------------
# LOG4CXX_LIBRARIES
#------------------------------------------------------------------------------

FIND_LIBRARY(LOG4CXX_LIBRARIES
    NAMES log4cxx
    DOC "Path to the log4cxx libraries."
    )

#------------------------------------------------------------------------------
# LOG4CXX_FOUND
#------------------------------------------------------------------------------

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LOG4CXX DEFAULT_MSG ${_SEARCHED_VARIABLES})
MARK_AS_ADVANCED(${_SEARCHED_VARIABLES})
