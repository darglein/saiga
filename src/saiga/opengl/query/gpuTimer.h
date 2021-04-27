/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/timeStampQuery.h"
#include "saiga/core/time/timer.h"
namespace Saiga
{
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
class SAIGA_OPENGL_API MultiFrameOpenGLTimer : public TimestampTimer
{
   public:
    MultiFrameOpenGLTimer(bool use_time_stamps = true);
    ~MultiFrameOpenGLTimer();

    /**
     * Creates the underlying OpenGL objects.
     */
    void create();

    void Start();
    void Stop();

    float getTimeMS();
    double getTimeMSd();
    uint64_t getTimeNS();

    std::pair<uint64_t, uint64_t> LastMeasurement() { return {begin_time, end_time}; }

   private:
    QueryObject queries[2][2];

    int queryBackBuffer = 0, queryFrontBuffer = 1;
    uint64_t elapsed_time = 0;
    uint64_t end_time     = 0;
    uint64_t begin_time   = 0;
    bool use_time_stamps;

    void swapQueries();
};

/**
 * Exponentially filtered GPUTimer.
 * time = alpha * newTime + (1-alpha) * oldTime;
 */

class SAIGA_OPENGL_API FilteredMultiFrameOpenGLTimer : public MultiFrameOpenGLTimer
{
   public:
    double alpha = 0.05;

    void stopTimer();
    float getTimeMS();
    double getTimeMSd();

   private:
    double currentTimeMS = 0;
};


class SAIGA_OPENGL_API OpenGLTimer
{
   public:
    OpenGLTimer();

    void start();
    GLuint64 stop();
    float getTimeMS();

   protected:
    QueryObject queries[2];
    GLuint64 time;
};


template <typename T>
class SAIGA_OPENGL_API ScopedOpenGLTimer : public OpenGLTimer
{
   public:
    T* target;
    ScopedOpenGLTimer(T* target) : target(target) { start(); }

    ScopedOpenGLTimer(T& target) : target(&target) { start(); }

    ~ScopedOpenGLTimer()
    {
        stop();
        T time  = static_cast<T>(getTimeMS());
        *target = time;
    }
};

class SAIGA_OPENGL_API ScopedOpenGLTimerPrint : public OpenGLTimer
{
   public:
    std::string name;
    ScopedOpenGLTimerPrint(const std::string& name);
    ~ScopedOpenGLTimerPrint();
};


}  // namespace Saiga
