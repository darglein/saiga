/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include "time.h"

#include <vector>

namespace Saiga
{
#ifdef WIN32
#    if _MSC_VER >= 1900  // VS2015 and newer
#        define HAS_HIGH_RESOLUTION_CLOCK
#    endif
#else
#    define HAS_HIGH_RESOLUTION_CLOCK
#endif

// typedef long long tick_t;
// typedef double tick_t;

// Linux: c++ 11 chrono for time measurement
// Windows: queryPerformanceCounter because c++ 11 chrono only since VS2015 with good precision :(
class SAIGA_CORE_API Timer
{
   public:
    Timer();
    virtual ~Timer() {}

    void start();
    tick_t stop();

    double getTimeMS();
    double getTimeMicrS();

    tick_t getTime() { return lastTime; }
    virtual tick_t getCurrentTime() { return getTime(); }

   protected:
    virtual void addMeassurment(tick_t time);
    //    double startTime;

    tick_t lastTime = tick_t(0);

#ifdef HAS_HIGH_RESOLUTION_CLOCK
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
#else
    int64_t startTime;
    int64_t ticksPerSecond;
    double freq = 0.0;
#endif
};


class SAIGA_CORE_API ExponentialTimer : public Timer
{
   public:
    ExponentialTimer(double alpha = 0.9);
    virtual tick_t getCurrentTime() override { return currentTime; }

   protected:
    virtual void addMeassurment(tick_t time) override;
    tick_t currentTime = tick_t(0);  // smoothed
    double alpha;
};


class SAIGA_CORE_API AverageTimer : public Timer
{
   public:
    AverageTimer(int number = 10);
    virtual tick_t getCurrentTime() override { return currentTime; }

    double getMinimumTimeMS();
    double getMaximumTimeMS();

   protected:
    virtual void addMeassurment(tick_t time) override;
    std::vector<tick_t> lastTimes;
    tick_t minimum     = tick_t(0);
    tick_t maximum     = tick_t(0);
    tick_t currentTime = tick_t(0);  // smoothed
    int currentTimeId  = 0;
    int number;
};

class SAIGA_CORE_API ScopedTimerPrint : public Timer
{
   public:
    std::string name;
    ScopedTimerPrint(const std::string& name);
    ~ScopedTimerPrint();
};


template <typename T = float, typename Unit = std::chrono::milliseconds>
class ScopedTimer : public Timer
{
    static_assert(std::is_arithmetic_v<T>);

   public:
    T* target;
    explicit ScopedTimer(T* target) : target(target) { start(); }

    explicit ScopedTimer(T& target) : target(&target) { start(); }

    ScopedTimer(ScopedTimer&& other) noexcept : target(other.target) {}


    ~ScopedTimer() override
    {
        stop();
        T time  = std::chrono::duration_cast<std::chrono::duration<T, typename Unit::period>>(getCurrentTime()).count();
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
