#include "saiga/util/timer2.h"
#include <GLFW/glfw3.h>
#include "saiga/util/glm.h"

Timer2::Timer2()
{

}

void Timer2::start()
{
    //glfwgettime returns time in seconds since initialization
//     startTime = glfwGetTime() * 1000;



    startTime = std::chrono::high_resolution_clock::now();
}

void Timer2::stop()
{
//    double elapsed = (glfwGetTime() * 1000) - startTime;
//    addMeassurment(elapsed);


    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = endTime - startTime;

    time_interval_t dt = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    addMeassurment(dt);
}

double Timer2::getTimeMS()
{
    return lastTime/1000.0;
}

double Timer2::getLastTimeMS()
{
    return lastTime/1000.0;
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
