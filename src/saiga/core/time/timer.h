/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

#include "TimerBase.h"
#include "time.h"

#include <vector>

namespace Saiga
{

// Interface for a simple TimestampTimer, which returns the start and end timestamps
// of the Start() and Stop() call. Note, that the timestamps should be in an absolute
// timeframe for example the program start.
class SAIGA_CORE_API TimestampTimer
{
   public:
    virtual ~TimestampTimer() {}
    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual std::pair<uint64_t, uint64_t> LastMeasurement()  = 0;
};

/**
 * In this file, more advanced timers are defined.
 */
class SAIGA_CORE_API Timer : public TimerBase
{
   public:
    Timer() { start(); }
    virtual ~Timer() {}

    void start() { startTime = std::chrono::high_resolution_clock::now(); }

    Time stop()
    {
        auto T = TimerBase::stop();
        addMeassurment(T);
        return T;
    }


    double getTimeMicrS()
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(getCurrentTime()).count();
    }

    double getTimeMS()
    {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(getCurrentTime()).count();
    }

    virtual tick_t getCurrentTime() { return getTime(); }

   protected:
    virtual void addMeassurment(Time time) { lastTime = time; }
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


}  // namespace Saiga
