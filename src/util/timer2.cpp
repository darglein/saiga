
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
	//Since VS2015 the standard high resolution clock is implemented with queryperformanceCounters,
	//so this special windows code is not needed anymore.
#ifdef WIN32
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	startTime = li.QuadPart;
#else
    startTime = std::chrono::high_resolution_clock::now();
#endif
}

void Timer2::stop()
{
#ifdef WIN32
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	time_interval_t dt = li.QuadPart - startTime;
	addMeassurment(dt);
#else
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = endTime - startTime;

    time_interval_t dt = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    addMeassurment(dt);
#endif
}

double Timer2::getTimeMS()
{
#ifdef WIN32
	return lastTime / PCFreq;
#else
    return lastTime/1000.0;
#endif
}

double Timer2::getLastTimeMS()
{
#ifdef WIN32
	return lastTime / PCFreq;
#else
    return lastTime/1000.0;
#endif
}

void Timer2::addMeassurment(time_interval_t time)
{
    lastTime = time;
}


//============================================================================================


ExponentialTimer::ExponentialTimer(double alpha) : alpha(alpha)
{

}

double ExponentialTimer::getTimeMS()
{
    return currentTimeMS;
}

void ExponentialTimer::addMeassurment(time_interval_t time)
{
    lastTime = time;
    currentTimeMS = alpha*currentTimeMS + (1-alpha)*getLastTimeMS();
}


//============================================================================================


AverageTimer::AverageTimer(int number) :  lastTimes(number,0), number(number)
{

}

double AverageTimer::getTimeMS()
{
    return currentTimeMS;
}

void AverageTimer::addMeassurment(time_interval_t time)
{
    lastTime = time;
    lastTimes[currentTimeId] = getLastTimeMS();
    currentTimeId = (currentTimeId+1) % lastTimes.size();

    currentTimeMS = 0;
    for(auto &d : lastTimes){
        currentTimeMS += d;
    }
    currentTimeMS /= lastTimes.size();
}

double AverageTimer::getMinimumTimeMS()
{
    double m = 93465943753535;
    for(double &d : lastTimes){
        m = glm::min(d,m);
    }
    return m;
}

double AverageTimer::getMaximumTimeMS()
{
    double m = -93465943753535;
    for(double &d : lastTimes){
        m = glm::max(d,m);
    }
    return m;
}
