/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/tostring.h"

#include <chrono>
#include <sstream>
#include <string>


namespace Saiga
{
using game_ratio_t = std::nano;
using tick_t       = std::chrono::duration<int64_t, game_ratio_t>;
using tickd_t      = std::chrono::duration<double, game_ratio_t>;

// using a floating point type here because we need to do alot of interpolation stuff
using animationtime_t = std::chrono::duration<double>;


// Time format: https://man7.org/linux/man-pages/man3/strftime.3.html
inline std::string CurrentTimeString(const std::string& format)
{
    const int b_size = 200;
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[b_size];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, b_size, format.c_str(), timeinfo);

    return std::string(buffer);
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
    strm << ":" << Saiga::leadingZeroString(ms, 4);
    return strm.str();
}
}  // namespace Saiga