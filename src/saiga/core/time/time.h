/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/tostring.h"

#include <chrono>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>


namespace Saiga
{
using game_ratio_t = std::nano;
using tick_t       = std::chrono::duration<int64_t, game_ratio_t>;
using tickd_t      = std::chrono::duration<double, game_ratio_t>;

// using a floating point type here because we need to do alot of interpolation stuff
using animationtime_t = std::chrono::duration<double>;


// Time format: https://man7.org/linux/man-pages/man3/strftime.3.html
// Example Format: "%F_%H-%M-%S"
inline std::string CurrentTimeString(const std::string& format)
{
    time_t rawtime;
    time(&rawtime);

#ifdef _WIN32
    struct tm timeinfo;
    localtime_s(&timeinfo, &rawtime);
#else
    // localtime is not thread-safe
    struct tm timeinfo;
    localtime_r(&rawtime, &timeinfo);
#endif

    // strftime returns 0 (and the buffer content is undefined) if the result does not fit into the buffer.
    // An empty result also returns 0, so give up at some point.
    for (size_t b_size = 200; b_size <= 200 + 64 * format.size(); b_size *= 2)
    {
        std::vector<char> buffer(b_size);
        size_t length = strftime(buffer.data(), b_size, format.c_str(), &timeinfo);
        if (length > 0) return std::string(buffer.data(), length);
    }
    return std::string();
}



template <typename DurationType>
inline std::string DurationToString(const DurationType& duration)
{
    // current time
    uint64_t h  = std::chrono::duration_cast<std::chrono::hours>(duration).count();
    uint64_t m  = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
    uint64_t s  = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

    std::stringstream strm;

    if (h > 0)
    {
        strm << Saiga::leadingZeroString(h, 2) << ":";
    }

    strm << Saiga::leadingZeroString(m, 2) << ":" << Saiga::leadingZeroString(s, 2);
    strm << ":" << Saiga::leadingZeroString(ms, 3);
    return strm.str();
}
}  // namespace Saiga