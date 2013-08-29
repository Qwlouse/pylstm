#include "Exceptions.h"

#include <execinfo.h> // for backtrace
#include <dlfcn.h> // for dladdr
#include <cxxabi.h> // for __cxa_demangle
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

#include <boost/exception/all.hpp>
namespace core
{

// from https://gist.github.com/fmela/591333
// This function produces a stack backtrace with demangled function & method names.
std::string get_backtrace(int skip = 1)
{
    void *callstack[128];
    const int nMaxFrames = 10;//sizeof(callstack) / sizeof(callstack[0]);
    char buf[1024];
    int nFrames = backtrace(callstack, nMaxFrames);
    char **symbols = backtrace_symbols(callstack, nFrames);
     
    std::ostringstream trace_buf;
    trace_buf << "Backtrace:\n";
    for (int i = skip; i < nFrames; i++) {
        printf("%s\n", symbols[i]);
         
        Dl_info info;
        if (dladdr(callstack[i], &info) && info.dli_sname) {
            char *demangled = NULL;
            int status = -1;
            if (info.dli_sname[0] == '_')
                demangled = abi::__cxa_demangle(info.dli_sname, NULL, 0, &status);
            snprintf(buf, sizeof(buf), "%-3d %*p %s + %zd\n", i, int(2 + sizeof(void*) * 2), callstack[i], status == 0 ? demangled : info.dli_sname == 0 ? symbols[i] : info.dli_sname, (char *)callstack[i] - (char *)info.dli_saddr);
            free(demangled);
        } else {
            snprintf(buf, sizeof(buf), "%-3d %*p %s\n", i, int(2 + sizeof(void*) * 2), callstack[i], symbols[i]);
        }
        trace_buf << buf;
    }
    free(symbols);
    if (nFrames == nMaxFrames)
        trace_buf << "[truncated]\n";
    return trace_buf.str();
}




// =================================== Exception ==============================
Exception::~Exception() throw()
{ }

const char* Exception::what() const throw()
{
    char const * diagInfo = NULL;
    try {
        (void) ::boost::exception_detail::diagnostic_information_impl(this, 0, false);
        diagInfo = ::boost::exception_detail::get_diagnostic_information(*this, 0);
    }
    catch( ... ) { }

    if (diagInfo != NULL)
        return diagInfo;
    else
        return "";
}

// =================================== RuntimeException =======================
RuntimeException::RuntimeException() throw()  :
    Exception(),
    std::runtime_error("DUMMY! Real message is stored as core::Throw_Message.")
{ 
    *this << Throw_Backtrace(get_backtrace(3));
}

const char* RuntimeException::what() const throw()
{
    return Exception::what();
}

// =================================== LogicErrorException ====================
LogicErrorException::LogicErrorException() throw()  :
    Exception(),
    std::logic_error("DUMMY! Real message is stored as core::Throw_Message.")
{ 
    *this << Throw_Backtrace(get_backtrace(3));
}

const char* LogicErrorException::what() const throw()
{
    return Exception::what();
}

// =================================== IllegalArgumentException ===============
IllegalArgumentException::IllegalArgumentException(const std::string& message) 
                                                                      throw() :
    RuntimeException()
{
    *this << Throw_Message(message);
}

// =================================== CommandLineException ===================
CommandLineException::CommandLineException(const std::string& message) 
                                                                      throw() :
    RuntimeException()
{
    *this << Throw_Message(message);
}

// =================================== AssertFailedException ==================
AssertFailedException::AssertFailedException(const std::string& message) 
                                                                      throw() :

    LogicErrorException()
{
    *this << Throw_Message(message);
}

// =================================== InvalidStateException ==================
InvalidStateException::InvalidStateException(const std::string& message) 
                                                                      throw() :
    LogicErrorException()
{
    *this << Throw_Message(message);
}

// =================================== InvalidStateException ==================
NotImplementedException::NotImplementedException(const std::string& message)
                                                                      throw() :
    LogicErrorException()
{
    *this << Throw_Message(message);
}

// =================================== Convienience Functions =================
std::ostream & operator<<(std::ostream& out, const Exception& e)
{
    out << e.what();
    return out;
}

}
