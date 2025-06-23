/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <iostream>
#include <cstdint>
#include <cstring>

struct NullStream : public std::ostream
{
    NullStream() : std::ostream(nullptr)
    {
    }
};


struct LogMessage
{
    LogMessage(const char* file, int line, bool result)
        : result(result)
    {
        stream() << "Check failed in " << file << ":" << line << "\n    ";
    }

    ~LogMessage()
    {
        if (!result)
        {
            stream() << std::endl;
            abort();
        }
    }

    static std::ostream& nstream()
    {
        static NullStream glog_nstrm;
        return glog_nstrm;
    }

    std::ostream& stream()
    {
        auto& strm = result ? nstream() : std::cout;
        return strm;
    }

    bool result;
};

#define CHECK_OP_LOG(name, op, val1, val2, log)             \
log(__FILE__, __LINE__, val1 op val2 ).stream()  \
<< "(" #val1 #op #val2 ") " << val1 << " " #op " " << val2 << "\n    "

#define CHECK_OP_LOG2(name,  val1, log)             \
    log(__FILE__, __LINE__, val1 ).stream()  \
    << "( " #val1 ")\n    "

#define CHECK_OP(name, op, val1, val2)  CHECK_OP_LOG(name, op, val1, val2, LogMessage)

#define CHECK(val1) CHECK_OP_LOG2(_EQ, (bool)(val1), LogMessage)

#define CHECK_NOTNULL(val1) CHECK_OP(_GT, != , val1, 0)
#define CHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(_LT, < , val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(_GT, > , val1, val2)


#define VLOG(x) std::cout