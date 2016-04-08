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



    startTime = std::chrono::system_clock::now();
}

void Timer2::stop()
{
//    double elapsed = (glfwGetTime() * 1000) - startTime;
//    addMeassurment(elapsed);

    auto endTime = std::chrono::system_clock::now();
    auto elapsed = endTime - startTime;

    double dt  = std::chrono::duration <double, std::milli> (elapsed).count();
    addMeassurment(dt);
}

double Timer2::getTimeMS()
{
    return lastTime;
}

double Timer2::getLastTimeMS()
{
    return lastTime;
}

void Timer2::addMeassurment(double time)
{
    lastTime = time;
}


//============================================================================================


ExponentialTimer::ExponentialTimer(double alpha) : alpha(alpha)
{

}

double ExponentialTimer::getTimeMS()
{
    return currentTime;
}

void ExponentialTimer::addMeassurment(double time)
{
    lastTime = time;
    currentTime = alpha*currentTime + (1-alpha)*time;
}


//============================================================================================


AverageTimer::AverageTimer(int number) :  lastTimes(number,0), number(number)
{

}

double AverageTimer::getTimeMS()
{
    return currentTime;
}

void AverageTimer::addMeassurment(double time)
{
    lastTime = time;
    lastTimes[currentTimeId] = time;
    currentTimeId = (currentTimeId+1) % lastTimes.size();

    currentTime = 0;
    for(double &d : lastTimes){
        currentTime += d;
    }
    currentTime /= lastTimes.size();
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
