/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/query/gpuTimer.h"

#include "saiga/core/util/assert.h"

#include <iostream>

namespace Saiga
{
MultiFrameOpenGLTimer::MultiFrameOpenGLTimer(bool use_time_stamps) : use_time_stamps(use_time_stamps) {}

MultiFrameOpenGLTimer::~MultiFrameOpenGLTimer() {}

void MultiFrameOpenGLTimer::create()
{
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            queries[i][j].create(use_time_stamps);
        }
    }
}

void MultiFrameOpenGLTimer::swapQueries()
{
    std::swap(queryBackBuffer, queryFrontBuffer);
}


void MultiFrameOpenGLTimer::Start()
{
    if (use_time_stamps)
        queries[queryBackBuffer][0].record();
    else
        queries[queryBackBuffer][0].begin();
}

void MultiFrameOpenGLTimer::Stop()
{
    if (use_time_stamps)
    {
        queries[queryBackBuffer][1].record();
        end_time     = queries[queryFrontBuffer][1].getTimestamp();
        begin_time   = queries[queryFrontBuffer][0].getTimestamp();
        elapsed_time = end_time - begin_time;
    }
    else
    {
        queries[queryBackBuffer][0].end();
        elapsed_time = queries[queryFrontBuffer][0].getTimestamp();
    }

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
    return elapsed_time;
}


//========================================================================


void FilteredMultiFrameOpenGLTimer::stopTimer()
{
    MultiFrameOpenGLTimer::Stop();
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
    queries[0].create(true);
    queries[1].create(true);
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
