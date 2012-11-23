# The core library
ADD_LIBRARY(core STATIC
            Exceptions.cpp
            Logging.cpp
            Assert.cpp)

IF(LOGGING_ENABLED)
    TARGET_LINK_LIBRARIES(core ${LOG4CXX_LIBRARIES})
ENDIF(LOGGING_ENABLED)
