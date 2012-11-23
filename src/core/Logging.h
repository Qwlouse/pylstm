#pragma once

#include <string>
#include "Config.h"
#ifdef LOGGING_ENABLED
#ifndef LOG_LEVEL
#   define LOG_LEVEL 4
#endif


#ifdef LOGGING_LOG4CXX
// ============================= Log4Cxx =======================================
// include log4cxx header files
#include <log4cxx/logger.h>

//! Should be called once per cpp file
#define ENABLE_LOGGING(name) \
        namespace { \
            log4cxx::LoggerPtr logger__(log4cxx::Logger::getLogger( name)); \
        } struct declaredToEatSemicolon
#if LOG_LEVEL >= 6
#    define LOG_TRACE(message) LOG4CXX_TRACE(logger__, message)
#endif
#if LOG_LEVEL >= 5
#    define LOG_DEBUG(message) LOG4CXX_DEBUG(logger__, message)
#endif
#if LOG_LEVEL >= 4
#    define LOG_INFO(message) LOG4CXX_INFO(logger__, message)
#endif
#if LOG_LEVEL >= 3
#    define LOG_WARN(message) LOG4CXX_WARN(logger__, message)
#endif
#if LOG_LEVEL >= 2
#    define LOG_ERROR(message) LOG4CXX_ERROR(logger__, message)
#endif
#if LOG_LEVEL >= 1
#    define LOG_FATAL(message) LOG4CXX_FATAL(logger__, message)
#endif


#elif defined LOGGING_CERR
// ============================= cerr logging ==================================
#include <iostream>

#define ENABLE_LOGGING(name) \
        namespace { \
            const char* logger__ = name; \
        } struct declaredToEatSemicolon

#if LOG_LEVEL >= 6
#    define LOG_TRACE(message)  (std::cerr << "TRACE " << logger__ << ": " << message << "\n")
#endif

#if LOG_LEVEL >= 5
#    define LOG_DEBUG(message)  (std::cerr << "DEBUG " << logger__ << ": " << message << "\n")
#endif

#if LOG_LEVEL >= 4
#     define LOG_INFO(message)  (std::cerr << "INFO "  << logger__ << ": " << message << "\n")
#endif

#if LOG_LEVEL >= 3
#     define LOG_WARN(message)  (std::cerr << "WARN "  << logger__ << ": " << message << "\n")
#endif

#if LOG_LEVEL >= 2
#     define LOG_ERROR(message) (std::cerr << "ERROR " << logger__ << ": " << message << "\n")
#endif

#if LOG_LEVEL >= 1
#     define LOG_FATAL(message) (std::cerr << "FATAL " << logger__ << ": " << message << "\n")
#endif

#endif // LOGGING_LOG4CXX

#endif // LOGGING_ENABLED

// ============================= no logging ====================================
// Now make sure all macros are defined no matter what happened
#ifndef ENABLE_LOGGING
#    define ENABLE_LOGGING(name) \
        namespace { \
            const char* logger__ = ""; \
        } struct declaredToEatSemicolon
#endif

#ifndef LOG_TRACE
#    define LOG_TRACE(message)   (void (logger__))
#endif

#ifndef LOG_DEBUG
#define LOG_DEBUG(message)       (void (logger__))
#endif

#ifndef LOG_INFO
#define LOG_INFO(message)        (void (logger__))
#endif

#ifndef LOG_WARN
#define LOG_WARN(message)        (void (logger__))
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(message)       (void (logger__))
#endif

#ifndef LOG_FATAL
#define LOG_FATAL(message)       (void (logger__))
#endif


//============================== Setup methods =================================
namespace core
{
//! should be called once per application to configure the logging
void setUpLogging();
void setUpLogging(const std::string& level);
}
