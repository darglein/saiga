/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/query/gpuTimer.h"

#include "saiga/core/util/assert.h"

#include <iostream>

namespace Saiga
{
MultiFrameOpenGLTimer::MultiFrameOpenGLTimer() {}

MultiFrameOpenGLTimer::~MultiFrameOpenGLTimer() {}

void MultiFrameOpenGLTimer::create()
{
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            queries[i][j].create();
        }
    }
}

void MultiFrameOpenGLTimer::swapQueries()
{
    std::swap(queryBackBuffer, queryFrontBuffer);
}


void MultiFrameOpenGLTimer::startTimer()
{
    queries[queryBackBuffer][0].record();
}

void MultiFrameOpenGLTimer::stopTimer()
{
    queries[queryBackBuffer][1].record();
    time = queries[queryFrontBuffer][1].getTimestamp() - queries[queryFrontBuffer][0].getTimestamp();

    //    time = queries[queryFrontBuffer][1].waitTimestamp() - queries[queryFrontBuffer][0].waitTimestamp();
    swapQueries();
}

float MultiFrameOpenGLTimer::getTimeMS()
{
    return getTimeNS() / 1000000.0f;
}

double MultiFrameOpenGLTimer::getTimeMSd()
{
    return getTimeNS() / 1000000.0;
}

GLuint64 MultiFrameOpenGLTimer::getTimeNS()
{
    return time;
}


//========================================================================


void FilteredMultiFrameOpenGLTimer::stopTimer()
{
    MultiFrameOpenGLTimer::stopTimer();
    double newTime = MultiFrameOpenGLTimer::getTimeMSd();
    currentTimeMS  = newTime * alpha + (1.0f - alpha) * currentTimeMS;
}

float FilteredMultiFrameOpenGLTimer::getTimeMS()
{
    return currentTimeMS;
}

double FilteredMultiFrameOpenGLTimer::getTimeMSd()
{
    return currentTimeMS;
}

OpenGLTimer::OpenGLTimer()
{
    queries[0].create();
    queries[1].create();
}

void OpenGLTimer::start()
{
    queries[0].record();
}

GLuint64 OpenGLTimer::stop()
{
    queries[1].record();
    time = queries[1].waitTimestamp() - queries[0].waitTimestamp();
    return time;
}

float OpenGLTimer::getTimeMS()
{
    return time / 1000000.0f;
}

ScopedOpenGLTimerPrint::ScopedOpenGLTimerPrint(const std::string& name) : name(name)
{
    start();
}

ScopedOpenGLTimerPrint::~ScopedOpenGLTimerPrint()
{
    stop();
    auto time = getTimeMS();
    std::cout << name << " : " << time << "ms." << std::endl;
}

}  // namespace Saiga
