/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/time/time.h"
#include <vector>

namespace Saiga {

#ifdef WIN32
#if _MSC_VER >= 1900 //VS2015 and newer
#define HAS_HIGH_RESOLUTION_CLOCK
#endif
#else
#define HAS_HIGH_RESOLUTION_CLOCK
#endif

//typedef long long tick_t;
//typedef double tick_t;

//Linux: c++ 11 chrono for time measurement
//Windows: queryPerformanceCounter because c++ 11 chrono only since VS2015 with good precision :(
class SAIGA_GLOBAL Timer{
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


class SAIGA_GLOBAL ExponentialTimer : public Timer{
public:
    ExponentialTimer(double alpha = 0.9);
    virtual tick_t getCurrentTime() override { return currentTime; }
protected:
    virtual void addMeassurment(tick_t time) override;
    tick_t currentTime = tick_t(0); //smoothed
    double alpha;
};


class SAIGA_GLOBAL AverageTimer : public Timer{
public:
    AverageTimer(int number = 10);
    virtual tick_t getCurrentTime() override { return currentTime; }

    double getMinimumTimeMS();
    double getMaximumTimeMS();
protected:
    virtual void addMeassurment(tick_t time) override;
    std::vector<tick_t> lastTimes;
    tick_t minimum = tick_t(0);
    tick_t maximum = tick_t(0);
    tick_t currentTime = tick_t(0); //smoothed
    int currentTimeId = 0;
    int number;
};

class SAIGA_GLOBAL ScopedTimerPrint : public Timer{
public:
    std::string name;
    ScopedTimerPrint(const std::string &name);
    ~ScopedTimerPrint();
};


template<typename T>
class ScopedTimer : public Timer{
public:
    T* target;
    ScopedTimer(T* target) : target(target){
        start();
    }

    ScopedTimer(T& target) : target(&target){
        start();
    }

    ~ScopedTimer(){
        stop();
        T time = static_cast<T>(getTimeMS());
        *target = time;
    }
};

}
