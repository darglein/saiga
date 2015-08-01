#include "saiga/opengl/query/gpuTimer.h"



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
//    glBeginQuery(GL_TIME_ELAPSED,queryBackBuffer);
    queries[queryBackBuffer][0].record();
}

void GPUTimer::stopTimer()
{
//    glEndQuery(GL_TIME_ELAPSED);
    queries[queryBackBuffer][1].record();

    time = queries[queryFrontBuffer][1].getTimestamp() - queries[queryFrontBuffer][0].getTimestamp();

//    glGetQueryObjectui64v(queryFrontBuffer,GL_QUERY_RESULT, &time);
    swapQueries();
}

float GPUTimer::getTimeMS()
{
    return time/1000000.0f;
}

GLuint64 GPUTimer::getTimeNS()
{
    return time;
}


