/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>



#if defined _WIN32 || defined __CYGWIN__
#    define TINY_GLOG_HELPER_DLL_IMPORT __declspec(dllimport)
#    define TINY_GLOG_HELPER_DLL_EXPORT __declspec(dllexport)
#    define TINY_GLOG_HELPER_DLL_LOCAL
#else
#    define TINY_GLOG_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#    define TINY_GLOG_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#    define TINY_GLOG_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#endif



#if defined(TINY_GLOG_STATIC)
#    define TINY_GLOG_API
#elif defined(tiny_glog_EXPORTS)
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

    // An active object calls the failure function in its destructor, so a copy would report the failure twice.
    LogStreamVoidifier(const LogStreamVoidifier&)            = delete;
    LogStreamVoidifier& operator=(const LogStreamVoidifier&) = delete;
    LogStreamVoidifier(LogStreamVoidifier&& other) noexcept : m_stream(other.m_stream) { other.m_stream = nullptr; }
    LogStreamVoidifier& operator=(LogStreamVoidifier&&) = delete;

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
    std::ostream* m_stream;  // Pointer to the actual stream (std::cerr or nullptr)
};


struct TINY_GLOG_API LogMessage
{
    LogMessage(const char* file, int line, bool result) : result(result)
    {
        if (!result)
        {
            // If check failed, do the initial output immediately to std::cerr
            std::cerr << "Check failed in " << file << ":" << line << "\n    ";
        }
    }

    // This method now returns our new proxy object by value.
    // This is the key to chaining and conditional evaluation.
    LogStreamVoidifier stream();

    bool result;
};

namespace tiny_glog
{
// Evaluates both operands once. On failure, prints the expression with both values and returns an active stream
// that calls the failure function when it is destroyed.
template <typename T1, typename T2, typename Compare>
LogStreamVoidifier CheckOp(const char* file, int line, const char* expression, const char* op, const T1& v1,
                           const T2& v2, Compare compare)
{
    if (compare(v1, v2))
    {
        return LogStreamVoidifier();
    }
    LogMessage message(file, line, false);
    std::cerr << expression << v1 << " " << op << " " << v2 << "\n    ";
    return message.stream();
}

template <typename T>
T CheckNotNull(const char* file, int line, const char* expression, T&& t)
{
    if (t == nullptr)
    {
        LogMessage message(file, line, false);
        message.stream() << "(" << expression << " != nullptr)\n";
    }
    return std::forward<T>(t);
}
}  // namespace tiny_glog

// The macros expand to 'if (...) {} else stream', so an 'else' that follows a CHECK in user code
// binds to the user's 'if' and not to the one inside the macro.
#define CHECK_OP(name, op, val1, val2)                                                                      \
    if (LogStreamVoidifier _ls_void =                                                                       \
            tiny_glog::CheckOp(__FILE__, __LINE__, "(" #val1 " " #op " " #val2 ") ", #op, (val1), (val2),   \
                               [](const auto& _a, const auto& _b) { return static_cast<bool>(_a op _b); }); \
        !_ls_void)                                                                                          \
    {                                                                                                       \
    }                                                                                                       \
    else                                                                                                    \
        _ls_void << ""

#define CHECK(val1)                                                                                     \
    if (LogStreamVoidifier _ls_void = LogMessage(__FILE__, __LINE__, (bool)(val1)).stream(); !_ls_void) \
    {                                                                                                   \
    }                                                                                                   \
    else                                                                                                \
        _ls_void << "( " #val1 ")\n    "

#define CHECK_NOTNULL(val1) tiny_glog::CheckNotNull(__FILE__, __LINE__, #val1, (val1))
#define CHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(_GT, >, val1, val2)

namespace tiny_glog
{
// One LOG(INFO/WARNING/ERROR) line. The line is collected and written with a single write when the temporary is
// destroyed at the end of the full expression, so lines of different threads don't interleave.
class LogLine
{
   public:
    LogLine(const char* severity, const char* file, int line) { buffer << severity << " " << file << ":" << line << "] "; }
    ~LogLine()
    {
        buffer << '\n';
        std::cerr << buffer.str() << std::flush;
    }
    LogLine(const LogLine&)            = delete;
    LogLine& operator=(const LogLine&) = delete;

    std::ostream& stream() { return buffer; }

   private:
    std::ostringstream buffer;
};
}  // namespace tiny_glog

// LOG(INFO), LOG(WARNING), LOG(ERROR) and LOG(FATAL) like glog. LOG(FATAL) prints the message and then calls the
// failure function (abort by default). The severity is pasted with ## and therefore not macro-expanded, so an ERROR
// macro (wingdi.h) does not break LOG(ERROR).
#define LOG(severity) TINY_GLOG_LOG_##severity
#define TINY_GLOG_LOG_INFO tiny_glog::LogLine("I", __FILE__, __LINE__).stream()
#define TINY_GLOG_LOG_WARNING tiny_glog::LogLine("W", __FILE__, __LINE__).stream()
#define TINY_GLOG_LOG_ERROR tiny_glog::LogLine("E", __FILE__, __LINE__).stream()
// An active LogStreamVoidifier ends the line and calls the failure function when it is destroyed.
#define TINY_GLOG_LOG_FATAL                                                     \
    if (LogStreamVoidifier _ls_void = LogStreamVoidifier(std::cerr); !_ls_void) \
    {                                                                           \
    }                                                                           \
    else                                                                        \
        _ls_void << "F " << __FILE__ << ":" << __LINE__ << "] "
#if defined(_WIN32)
// wingdi.h defines ERROR as 0, which is expanded before the ## when LOG is called through another macro.
#    define TINY_GLOG_LOG_0 TINY_GLOG_LOG_ERROR
#endif

// Like glog with the default verbosity 0: only VLOG(0) prints.
#define VLOG(x)  \
    if ((x) > 0) \
    {            \
    }            \
    else         \
        std::cerr


namespace google
{
TINY_GLOG_API void InstallFailureFunction(void (*custom_glog_fail_func)());
}  // namespace google