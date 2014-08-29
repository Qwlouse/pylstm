#include <string>

#include <gtest/gtest.h>

#include "Config.h"
// ignore possible deactivation of asserts
#undef DISABLE_ASSERTS
#include "core/Assert.h"
#include "core/Exceptions.h"


using namespace core;
using std::string;

TEST(TestAssertions, ASSERT_true_noThrow)
{
    ASSERT_NO_THROW(ASSERT(true));
}

TEST(TestAssertions, ASSERT_false_throwAssertFailedException)
{
    ASSERT_THROW(ASSERT(false), AssertFailedException );
}

TEST(TestAssertions, ASSERT_false_exceptionContainsCorrectLine)
{
    int expectedLine;
    int* recoveredLine = NULL;
    try
    {
        expectedLine = __LINE__ + 1;
        ASSERT(false);
    } catch (AssertFailedException& e)
    {
        recoveredLine = boost::get_error_info<boost::throw_line>(e);
    }
    ASSERT_FALSE(recoveredLine == NULL);
    ASSERT_EQ(expectedLine, *recoveredLine);
}

TEST(TestAssertions, ASSERT_false_exceptionContainsCorrectFile)
{
    string expectedFilename(__FILE__);
    const char** recoveredFilename = NULL;
    try
    {
        ASSERT(false);
    } catch (AssertFailedException& e)
    {
        recoveredFilename = boost::get_error_info<boost::throw_file>(e);
    }
    ASSERT_FALSE(recoveredFilename == NULL);
    ASSERT_EQ(0, expectedFilename.compare(*recoveredFilename));
}

TEST(TestAssertions, ASSERT_false_exceptionContainsCorrectFunctionName)
{
    string expectedFunctionName(BOOST_CURRENT_FUNCTION);
    const char** recoveredFunctionName = NULL;
    try
    {
        ASSERT(false);
    } catch (AssertFailedException& e)
    {
        recoveredFunctionName = boost::get_error_info<boost::throw_function>(e);
    }
    ASSERT_FALSE(recoveredFunctionName == NULL);
    ASSERT_EQ(0, expectedFunctionName.compare(*recoveredFunctionName));
}

TEST(TestAssertions, ASSERT_false_exceptionContainsAssertStatement)
{
    string expectedStatement("false");
    const char** recoveredStatement;
    try
    {
        ASSERT(false);
    } catch (AssertFailedException& e)
    {
        recoveredStatement = boost::get_error_info<core::Throw_Expression>(e);
        ASSERT_FALSE(recoveredStatement == NULL);
        ASSERT_EQ(0, expectedStatement.compare(*recoveredStatement));
    }
    ASSERT_FALSE(recoveredStatement == NULL);
}

TEST(TestAssertions, ASSERT_compoundCondition_noThrow)
{
    ASSERT_NO_THROW(ASSERT(true || false));
    ASSERT_NO_THROW(ASSERT(false || true));
}

TEST(TestAssertions, ASSERT_withinIfStatement_elseShouldNotReferToAssert)
{
    bool wasElse = false;
    if (false)
        ASSERT(true);
    else
        wasElse = true;
    ASSERT_TRUE(wasElse);
}

