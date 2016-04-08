#pragma once

#include "saiga/config.h"
#include <vector>
#include <chrono>

//uses c++ 11 chrono for time meassuerment
class SAIGA_GLOBAL Timer2{
public:
    Timer2();

    void start();
    void stop();

    virtual double getTimeMS();
    double getLastTimeMS();
protected:
    virtual void addMeassurment(double time);
//    double startTime;
    std::chrono::time_point<std::chrono::system_clock> startTime;
    double lastTime;
};


class SAIGA_GLOBAL ExponentialTimer : public Timer2{
public:
    ExponentialTimer(double alpha = 0.9);
    virtual double getTimeMS() override;
protected:
    virtual void addMeassurment(double time) override;
    double currentTime = 0.0; //smoothed
    double alpha;
};


class SAIGA_GLOBAL AverageTimer : public Timer2{
public:
    AverageTimer(int number = 10);
    virtual double getTimeMS() override;

    double getMinimumTimeMS();
    double getMaximumTimeMS();
protected:
    virtual void addMeassurment(double time) override;
    std::vector<double> lastTimes;
    int currentTimeId = 0;
    double currentTime = 0.0; //smoothed
    int number;
};
