/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/timeStampQuery.h"

namespace Saiga {

/**
 * Asynchronous OpenGL GPU timer.
 *
 * Meassuers the time of the OpenGL calls between startTimer and stopTimer.
 * These calls do not empty the GPU command queue and return immediately.
 *
 * startTimer and stopTimer can only be called once per frame.
 * startTimer and stopTimer from multiple GPUTimers CAN be interleaved.
 *
 * The difference between GPUTimer and TimerQuery is, that GPUTimer uses internally GL_TIMESTAMP queries,
 * while TimerQuery uses GL_TIME_ELAPSED queries. GL_TIME_ELAPSED queries can not be interleaved, so
 * GPUTimer is the recommended way of OpenGL performance meassurment.
 *
 * Core since version 	3.3
 *
 */
class SAIGA_GLOBAL GPUTimer{
private:

    TimeStampQuery queries[2][2];

    int queryBackBuffer=0,queryFrontBuffer=1;
    GLuint64 time;

    void swapQueries();

#ifdef SAIGA_DEBUG
    bool stopped = false;
#endif
public:
    GPUTimer();
    ~GPUTimer();

    /**
     * Creates the underlying OpenGL objects.
     */
    void create();

    void startTimer();
    void stopTimer();

    float getTimeMS();
    double getTimeMSd();
    GLuint64 getTimeNS();
};

/**
 * Exponentially filtered GPUTimer.
 * time = alpha * newTime + (1-alpha) * oldTime;
 */

class SAIGA_GLOBAL FilteredGPUTimer : public GPUTimer{
private:
    double currentTimeMS = 0;
public:
    double alpha = 0.05;

    void stopTimer();
    float getTimeMS();
    double getTimeMSd();

};

}
