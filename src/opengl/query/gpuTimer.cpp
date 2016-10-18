#include "saiga/opengl/query/gpuTimer.h"
#include <algorithm> 
#include "saiga/util/assert.h"

GPUTimer::GPUTimer()
{
}

GPUTimer::~GPUTimer()
{
}

void GPUTimer::create()
{
    for(int i = 0 ; i < 2 ; ++i){
        for(int j = 0 ; j < 2 ; ++j){
            queries[i][j].create();
        }
    }
}

void GPUTimer::swapQueries()
{
    std::swap(queryBackBuffer,queryFrontBuffer);
}


void GPUTimer::startTimer()
{
    queries[queryBackBuffer][0].record();
}

void GPUTimer::stopTimer()
{
    queries[queryBackBuffer][1].record();
    time = queries[queryFrontBuffer][1].getTimestamp() - queries[queryFrontBuffer][0].getTimestamp();

//    time = queries[queryFrontBuffer][1].waitTimestamp() - queries[queryFrontBuffer][0].waitTimestamp();
    swapQueries();

#ifdef SAIGA_DEBUG
    stopped = true;
#endif
}

float GPUTimer::getTimeMS()
{
    return getTimeNS()/1000000.0f;
}

double GPUTimer::getTimeMSd()
{
    return getTimeNS()/1000000.0;
}

GLuint64 GPUTimer::getTimeNS()
{
#ifdef SAIGA_DEBUG
    assert(stopped && "GPU timer read before it was stopped once, time is not yet initialized");
#endif
    return time;
}


//========================================================================


void FilteredGPUTimer::stopTimer()
{
    GPUTimer::stopTimer();
    double newTime = GPUTimer::getTimeMSd();
    currentTimeMS = newTime*alpha + (1.0f-alpha) * currentTimeMS;
}

float FilteredGPUTimer::getTimeMS()
{
    return currentTimeMS;
}

double FilteredGPUTimer::getTimeMSd()
{
    return currentTimeMS;
}
