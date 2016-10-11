
#ifdef WIN32
#include <windows.h>
#endif

#include "saiga/util/timer2.h"
#include <GLFW/glfw3.h>
#include "saiga/util/glm.h"


Timer2::Timer2()
{

}

void Timer2::start()
{

#ifdef HAS_HIGH_RESOLUTION_CLOCK
    startTime = std::chrono::high_resolution_clock::now();
#else
    //Since VS2015 the standard high resolution clock is implemented with queryperformanceCounters,
    //so this special windows code is not needed anymore.
    LARGE_INTEGER li;
    if (!QueryPerformanceFrequency(&li))
        cout << "QueryPerformanceFrequency failed!\n";

//    cout << "QueryPerformanceFrequency " << double(li.QuadPart) << endl;
//    PCFreqPerMicrSecond = double(li.QuadPart) / 1000000.0;

    ticksPerSecond = li.QuadPart;

    assert(ticksPerSecond >= gameTime.base.count());

    freq = (double)ticksPerSecond / gameTime.base.count();

    QueryPerformanceCounter(&li);
    startTime = li.QuadPart;
#endif
}

void Timer2::stop()
{
#ifdef HAS_HIGH_RESOLUTION_CLOCK
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = endTime - startTime;
    tick_t dt = std::chrono::duration_cast<tick_t>(elapsed);
    addMeassurment(dt);
#else
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    double dt = (li.QuadPart - startTime) / freq;
    std::chrono::duration<double,game_ratio_t> dtc = dt;
    addMeassurment( std::chrono::duration_cast<tick_t>(dtc));
#endif
}


double Timer2::getTimeMicrS()
{
   return std::chrono::duration_cast<std::chrono::duration<double,std::micro>> (getCurrentTime()).count();
}

double Timer2::getTimeMS()
{
    return std::chrono::duration_cast<std::chrono::duration<double,std::milli>> (getCurrentTime()).count();
}


void Timer2::addMeassurment(tick_t time)
{
    lastTime = time;
}


//============================================================================================


ExponentialTimer::ExponentialTimer(double alpha) : alpha(alpha)
{

}

void ExponentialTimer::addMeassurment(tick_t time)
{
    lastTime = time;
    currentTime = std::chrono::duration_cast<tick_t> (alpha*currentTime + (1-alpha)*time);
}


//============================================================================================


AverageTimer::AverageTimer(int number) :  lastTimes(number,tick_t(0)), number(number)
{

}



void AverageTimer::addMeassurment(tick_t time)
{
    lastTime = time;
    lastTimes[currentTimeId] = time;
    currentTimeId = (currentTimeId+1) % lastTimes.size();

    currentTime = tick_t(0);
    minimum = lastTimes[0];
    maximum = lastTimes[0];
    for(auto &d : lastTimes){
        currentTime += d;
        minimum = std::min(minimum,d);
        maximum = std::max(maximum,d);
    }
    currentTime /= lastTimes.size();
}

double AverageTimer::getMinimumTimeMS()
{
    return std::chrono::duration_cast<std::chrono::duration<double,std::milli>> (minimum).count();
}

double AverageTimer::getMaximumTimeMS()
{
    return std::chrono::duration_cast<std::chrono::duration<double,std::milli>> (maximum).count();
}
