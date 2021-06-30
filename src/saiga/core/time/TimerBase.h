/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
namespace Saiga
{
#ifdef _WIN32
#    if _MSC_VER < 1900  // VS2015 and newer
#        error Your compiler is too old.
#    endif
#endif

/**
 * A very simple C++11 timer.
 */
class SAIGA_TEMPLATE TimerBase
{
   public:
    // 64-bit nanoseconds
    using Time = std::chrono::duration<int64_t, std::nano>;

    TimerBase() { start(); }

    void start() { startTime = std::chrono::high_resolution_clock::now(); }

    Time stop()
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsed = endTime - startTime;
        lastTime     = std::chrono::duration_cast<Time>(elapsed);
        return lastTime;
    }


    double getTimeMicrS()
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(lastTime).count();
    }

    double getTimeMS()
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(lastTime).count();
    }
    Time getTime() { return lastTime; }

   protected:
    Time lastTime = Time(0);
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

/**
 * Simple Scoped timer which prints the time in Milliseconds after desctruction.
 */
class SAIGA_TEMPLATE ScopedTimerPrint : public TimerBase
{
   public:
    ScopedTimerPrint(const std::string& name) : name(name) { start(); }
    ~ScopedTimerPrint()
    {
        stop();
        auto time = getTimeMS();
        std::cout << name << " : " << time << "ms." << std::endl;
    }

   public:
    std::string name;
};

class SAIGA_TEMPLATE ScopedTimerPrintLine : public TimerBase
{
   public:
    ScopedTimerPrintLine(const std::string& name) : name(name)
    {
        std::cout << "Starting " << name << "..." << std::flush;
        start();
    }
    ~ScopedTimerPrintLine()
    {
        stop();
        auto time = getTimeMS();
        std::cout << " Done in " << time << "ms." << std::endl;
    }

   public:
    std::string name;
};


template <typename T = float, typename Unit = std::chrono::milliseconds>
class ScopedTimer : public TimerBase
{
    static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");

   public:
    T* target;
    explicit ScopedTimer(T* target) : target(target) { start(); }

    explicit ScopedTimer(T& target) : target(&target) { start(); }

    ScopedTimer(ScopedTimer&& other) noexcept : target(other.target) {}
    ~ScopedTimer()
    {
        T time  = std::chrono::duration_cast<std::chrono::duration<T, typename Unit::period>>(stop()).count();
        *target = time;
    }
};

template <typename Unit = std::chrono::milliseconds, typename T = double>
auto make_scoped_timer(T& target)
{
    return ScopedTimer<T, Unit>(target);
}
}  // namespace Saiga


#define SAIGA_BLOCK_TIMER_NOMSG()                                                  \
    Saiga::ScopedTimerPrint __func_timer(std::string(SAIGA_SHORT_FUNCTION) + ":" + \
                                         std::string(std::to_string(__LINE__)))

#define SAIGA_BLOCK_TIMER_MSG(_msg)                                                                          \
    Saiga::ScopedTimerPrint __func_timer(std::string(_msg) + " " + std::string(SAIGA_SHORT_FUNCTION) + ":" + \
                                         std::string(std::to_string(__LINE__)))

#ifdef _MSC_VER
// The macro overloading with 0 arguments doesn't work with MSVC.
// -> Just use the normal timer without message.
#    define SAIGA_BLOCK_TIMER(...) SAIGA_BLOCK_TIMER_NOMSG()

#else


#    define GET_SAIGA_BLOCK_TIMER_MACRO(_0, _1, NAME, ...) NAME
#    define SAIGA_BLOCK_TIMER(...) \
        GET_SAIGA_BLOCK_TIMER_MACRO(_0, ##__VA_ARGS__, SAIGA_BLOCK_TIMER_MSG, SAIGA_BLOCK_TIMER_NOMSG)(__VA_ARGS__)

#endif

#define SAIGA_OPTIONAL_BLOCK_TIMER(_condition)                                                                       \
    auto __op_func_timer = (_condition)                                                                              \
                               ? std::make_shared<Saiga::ScopedTimerPrint>(std::string(SAIGA_SHORT_FUNCTION) + ":" + \
                                                                           std::string(std::to_string(__LINE__)))    \
                               : nullptr
