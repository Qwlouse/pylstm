#include "Logging.h"
#include <string>

using std::string;

#ifdef LOGGING_ENABLED
#ifdef LOGGING_LOG4CXX
#include <log4cxx/xml/domconfigurator.h>
void core::setUpLogging()
{
    string configFileName = string(PROJECT_CONFIG_DIR) + string("/logging_default.xml");
    log4cxx::xml::DOMConfigurator::configure(configFileName);
}

void core::setUpLogging(const string& level)
{
    string configFileName = string(PROJECT_CONFIG_DIR) + string("/logging_default.xml");
    log4cxx::xml::DOMConfigurator::configure(configFileName);
    log4cxx::Logger::getRootLogger()->setLevel(log4cxx::Level::toLevel(level));
}
#endif // LOGGING_LOG4CXX
#else
void core::setUpLogging() { }
void core::setUpLogging(const string&) { }

#endif // LOGGING_ENABLED
