// Copyright 2011 by Klaus Greff.
// All rights reserved.
#include <iostream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <gtest/gtest-typed-test.h>

#include "Core.h"


using std::string;
using namespace core;

// ================================= Set up Parameterized Test =================
template<typename T>
class TestExceptions : public::testing::Test
{
public:
    TestExceptions() :
        message_("TestMessage"),
        exceptionWithMessage_(new T(message_))
    {   }

    virtual ~TestExceptions()
    {
        delete exceptionWithMessage_;
    }

    // This is for the current type to be accessible via "typename TestFixture::ExceptionType"
    // The typename keyword is needed to satisfy the compiler
    typedef T ExceptionType;

    string message_;
    T* exceptionWithMessage_;

    typedef boost::error_info<struct throwCustomData, int> Throw_CustomData;

private:
    DISALLOW_COPY_AND_ASSIGN(TestExceptions);
};

#if GTEST_HAS_TYPED_TEST

using::testing::Types;

typedef Types<IllegalArgumentException, AssertFailedException, InvalidStateException> Implementations;

TYPED_TEST_CASE(TestExceptions, Implementations);


// ================================= Constructor ===============================

TYPED_TEST(TestExceptions, constructor_message_noThrow)
{
    ASSERT_NO_THROW(typename TestFixture::ExceptionType("Any message"));
}

TYPED_TEST(TestExceptions, copyConstructor_otherException_noThrow)
{
    ASSERT_NO_THROW(typename TestFixture::ExceptionType test(*TestFixture::exceptionWithMessage_));
}

TYPED_TEST(TestExceptions, constructor_message_isPartOfWhat)
{
    string whatString = string(TestFixture::exceptionWithMessage_->what());
    ASSERT_NE(whatString.find(TestFixture::message_), string::npos);
}

// ================================= Copy Constructor & Assignment ============

TYPED_TEST(TestExceptions, copyConstructor_otherException_messagesAreTheSame)
{
    typename TestFixture::ExceptionType copy(*this->exceptionWithMessage_);
    string message(this->exceptionWithMessage_->what());
    ASSERT_TRUE(message.compare(copy.what()) == 0);
}

TYPED_TEST(TestExceptions, assignment_otherException_messagesAreTheSame)
{
    typename TestFixture::ExceptionType copy = *this->exceptionWithMessage_;
    string message(this->exceptionWithMessage_->what());
    ASSERT_TRUE(message.compare(copy.what()) == 0);
}

// ================================= Stream Operator ===========================

TYPED_TEST(TestExceptions, streamOperator_LineNumber_canDirectlyBeRecovered)
{
    *this->exceptionWithMessage_ << ::boost::throw_line(17);
    int* line = boost::get_error_info<boost::throw_line>(*this->exceptionWithMessage_);
    ASSERT_EQ(17, *line);
}

TYPED_TEST(TestExceptions, streamOperator_filename_canDirectlyBeRecovered)
{
    const char* filename = "yourFilenameHere";
    *this->exceptionWithMessage_ << ::boost::throw_file(filename);
    const char** recoveredFilename = boost::get_error_info<boost::throw_file>(*this->exceptionWithMessage_);
    ASSERT_EQ(filename, *recoveredFilename);
}

TYPED_TEST(TestExceptions, streamOperator_functionName_canDirectlyBeRecovered)
{
    const char* functionName = "yourFunctionNameHere";
    *this->exceptionWithMessage_ << ::boost::throw_function(functionName);
    const char** recoveredFunctionName = boost::get_error_info<boost::throw_function>(*this->exceptionWithMessage_);
    ASSERT_EQ(functionName, *recoveredFunctionName);
}

TYPED_TEST(TestExceptions, streamOperator_customData_canDirectlyBeRecovered)
{

    *this->exceptionWithMessage_ << typename TestFixture::Throw_CustomData(42);
    int* recoveredCustomData = boost::get_error_info<typename TestFixture::Throw_CustomData>(
        *this->exceptionWithMessage_);
    ASSERT_EQ(42, *recoveredCustomData);
}

TYPED_TEST(TestExceptions, streamOperator_customData_canBeAppended)
{
    try
    {
        try
        {
            THROW(*this->exceptionWithMessage_);
        }
        catch (Exception& e)
        {
            e << typename TestFixture::Throw_CustomData(47);
            throw;
        }
    }
    catch (Exception& e)
    {
        int* recoveredCustomData = boost::get_error_info<typename TestFixture::Throw_CustomData>(e);
        ASSERT_EQ(47, *recoveredCustomData);
    }
}

// ================================= THROW Macro ===============================

TYPED_TEST(TestExceptions, THROW_MessageCanBeRecovered)
{
    try
    {
        THROW(*this->exceptionWithMessage_);
    }
    catch (Exception& e)
    {
        const string* message = boost::get_error_info<Throw_Message>(e);
        ASSERT_TRUE(message->compare(TestFixture::message_) == 0);
    }
}

TYPED_TEST(TestExceptions, THROW_LineCanBeRecovered)
{
    int expectedLine;
    try
    {
        expectedLine = __LINE__ + 1;
        THROW(*this->exceptionWithMessage_);
    }
    catch (Exception& e)
    {
        int* line = boost::get_error_info<boost::throw_line>(e);
        ASSERT_EQ(expectedLine, *line);
    }
}

TYPED_TEST(TestExceptions, THROW_FilenameCanBeRecovered)
{
    string filename = string(__FILE__);
    try
    {
        THROW(*this->exceptionWithMessage_);
    }
    catch (Exception& e)
    {
        const char** recoveredFilename = boost::get_error_info<boost::throw_file>(e);
        ASSERT_EQ(0, filename.compare(*recoveredFilename));
    }
}

TYPED_TEST(TestExceptions, THROW_FunctionNameCanBeRecovered)
{
    string functionName = string(BOOST_CURRENT_FUNCTION);
    try
    {
        THROW(*this->exceptionWithMessage_);
    }
    catch (boost::exception& e)
    {
        const char** recoveredFunctionName = boost::get_error_info<boost::throw_function>(e);
        ASSERT_EQ(0, functionName.compare(*recoveredFunctionName));
    }
}

TYPED_TEST(TestExceptions, THROW_CustomDataCanBeRecovered)
{
    try
    {
        THROW(*this->exceptionWithMessage_ << typename TestFixture::Throw_CustomData(42));
    }
    catch (boost::exception& e)
    {
        int* recoveredCustomData = boost::get_error_info<typename TestFixture::Throw_CustomData>(e);
        ASSERT_EQ(42, *recoveredCustomData);
    }
}

#endif // GTEST_HAS_TYPED_TEST
