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
class SAIGA_GLOBAL MultiFrameOpenGLTimer{
private:

    TimeStampQuery queries[2][2];

    int queryBackBuffer=0,queryFrontBuffer=1;
    GLuint64 time = 0;

    void swapQueries();
public:
    MultiFrameOpenGLTimer();
    ~MultiFrameOpenGLTimer();

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

class SAIGA_GLOBAL FilteredMultiFrameOpenGLTimer : public MultiFrameOpenGLTimer{
private:
    double currentTimeMS = 0;
public:
    double alpha = 0.05;

    void stopTimer();
    float getTimeMS();
    double getTimeMSd();

};


class SAIGA_GLOBAL OpenGLTimer{
protected:
    TimeStampQuery queries[2];
    GLuint64 time;
public:
    OpenGLTimer();

    void start();
    GLuint64 stop();
    float getTimeMS();
};


template<typename T>
class SAIGA_GLOBAL ScopedOpenGLTimer : public OpenGLTimer{
public:
    T* target;
    ScopedOpenGLTimer(T* target) : target(target){
        start();
    }

    ScopedOpenGLTimer(T& target) : target(&target){
        start();
    }

    ~ScopedOpenGLTimer(){
        stop();
        T time = static_cast<T>(getTimeMS());
        *target = time;
    }
};

class SAIGA_GLOBAL ScopedOpenGLTimerPrint : public OpenGLTimer{
public:
    std::string name;
    ScopedOpenGLTimerPrint(const std::string &name);
    ~ScopedOpenGLTimerPrint();
};


}
