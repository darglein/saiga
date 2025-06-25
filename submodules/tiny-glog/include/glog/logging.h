/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>



#if defined _WIN32 || defined __CYGWIN__
#    define TINY_GLOG_HELPER_DLL_IMPORT __declspec(dllimport)
#    define TINY_GLOG_HELPER_DLL_EXPORT __declspec(dllexport)
#    define TINY_GLOG_HELPER_DLL_LOCAL
#else
#    define TINY_GLOG_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#    define TINY_GLOG_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#    define TINY_GLOG_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#endif



#ifdef tiny_glog_EXPORTS
#    define TINY_GLOG_API TINY_GLOG_HELPER_DLL_EXPORT
#else
#    define TINY_GLOG_API TINY_GLOG_HELPER_DLL_IMPORT
#endif



// --- The New Conditional Proxy Object ---
// This object will be returned by LogMessage::stream()
// It provides an operator<< that conditionally writes,
// AND an implicit conversion to bool to enable the if (...) check.
class TINY_GLOG_API LogStreamVoidifier
{
   public:
    // Constructor for the active (logging enabled) state
    LogStreamVoidifier(std::ostream& os) : m_stream(&os) {}

    // Constructor for the inactive (logging disabled) state
    LogStreamVoidifier() : m_stream(nullptr) {}

    ~LogStreamVoidifier();

    // Overload for operator<< that only writes if m_stream is not null
    template <typename T>
    LogStreamVoidifier& operator<<(const T& val)
    {
        if (m_stream)
        {
            *m_stream << val;
        }
        return *this;
    }

    // Handle manipulators like std::endl, std::flush
    LogStreamVoidifier& operator<<(std::ostream& (*pf)(std::ostream&))
    {
        if (m_stream)
        {
            *m_stream << pf;
        }
        return *this;
    }

    // Crucial: Implicit conversion to bool.
    // This allows the "if (LogMessage(...).stream())" pattern.
    // When m_stream is nullptr, it evaluates to false, short-circuiting the "&&"
    // or allowing an if statement to skip the block.
    operator bool() const { return m_stream != nullptr; }

   private:
    std::ostream* m_stream;  // Pointer to the actual stream (std::cout or nullptr)
};


struct TINY_GLOG_API LogMessage
{
    LogMessage(const char* file, int line, bool result) : result(result)
    {
        if (!result)
        {
            // If check failed, do the initial output immediately to std::cout
            std::cout << "Check failed in " << file << ":" << line << "\n    ";
        }
    }

    // This method now returns our new proxy object by value.
    // This is the key to chaining and conditional evaluation.
    LogStreamVoidifier stream();

    bool result;
};

// --- Modified Macros ---

// The key change is to use an 'if' statement that leverages the 'operator bool()'
// of the LogStreamVoidifier.
// The `(void)0` provides an empty statement if the if condition is false.
// This is a common trick to make the macro parse correctly in all contexts.
#define CHECK_OP_LOG(name, op, val1, val2, log)                                       \
    if (LogStreamVoidifier _ls_void = log(__FILE__, __LINE__, val1 op val2).stream()) \
    _ls_void << "(" #val1 #op #val2 ") " << val1 << " " #op " " << val2 << "\n    "

#define CHECK_OP_LOG2(name, val1, log) \
    if (LogStreamVoidifier _ls_void = log(__FILE__, __LINE__, val1).stream()) _ls_void << "( " #val1 ")\n    "


// Keep the rest of your macros as they are, as they use CHECK_OP_LOG / CHECK_OP_LOG2
#define CHECK_OP(name, op, val1, val2) CHECK_OP_LOG(name, op, val1, val2, LogMessage)
#define CHECK(val1) CHECK_OP_LOG2(_EQ, (bool)(val1), LogMessage)
#define CHECK_NOTNULL(val1) CHECK_OP(_GT, !=, val1, 0)
#define CHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(_GT, >, val1, val2)

#define VLOG(x) std::cout


namespace google
{
TINY_GLOG_API void InstallFailureFunction(void (*custom_glog_fail_func)());
}  // namespace google