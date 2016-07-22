#pragma once

#include "saiga/config.h"
#include <vector>
#include <chrono>

typedef long long time_interval_t;
//typedef double time_interval_t;

//Linux: c++ 11 chrono for time measurement
//Windows: queryPerformanceCounter because c++ 11 chrono only since VS2015 with good precision :(
class SAIGA_GLOBAL Timer2{
public:
    Timer2();

    void start();
    void stop();

    virtual double getTimeMS();
    double getLastTimeMS();
    double getTimeMicrS();
protected:
    virtual void addMeassurment(time_interval_t time);
//    double startTime;
    
    time_interval_t lastTime;

#ifdef WIN32
	time_interval_t startTime;
    double PCFreqPerMicrSecond = 0.0;
#else
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
#endif
};


class SAIGA_GLOBAL ExponentialTimer : public Timer2{
public:
    ExponentialTimer(double alpha = 0.9);
    virtual double getTimeMS() override;
protected:
    virtual void addMeassurment(time_interval_t time) override;
    double currentTimeMS = 0.0; //smoothed
    double alpha;
};


class SAIGA_GLOBAL AverageTimer : public Timer2{
public:
    AverageTimer(int number = 10);
    virtual double getTimeMS() override;

    double getMinimumTimeMS();
    double getMaximumTimeMS();
protected:
    virtual void addMeassurment(time_interval_t time) override;
    std::vector<double> lastTimes;
    int currentTimeId = 0;
    double currentTimeMS = 0.0; //smoothed
    int number;
};
