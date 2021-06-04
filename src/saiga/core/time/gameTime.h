/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include "timer.h"

#include <chrono>

namespace Saiga
{
class SAIGA_CORE_API GameTime
{
   private:
    Timer gameTimer;
    double timeScale = 1.f;
    tick_t nextUpdateTime;
    tick_t nextFrameTime;

    // realtime scaled by timeScale
    tick_t scaledTime = tick_t(0);

    // only used for interpolation calculation
    tick_t actualUpdateTime;

    // time since start of the game
    tick_t lastRealTime;
    tick_t realTime;

    // when updating this is equal to update time and while rendering this is equal to render time
    tick_t currentTime;

    bool jtl = false;

   public:
    tick_t base = std::chrono::seconds(1);


    tick_t maxGameLoopDelay;


    // time at which the last 'update' took place
    // while updating this is equal to time
    // while rendering time should be greater than updatetime
    tick_t updatetime;

    tick_t renderTime;
    double interpolation;

    // timestep of 'update'
    tick_t dt;

    // timestep of 'render' (only != 0 if fps are limited)
    tick_t dtr;

    bool printInfoMsg = true;

    void init(tick_t _dt, tick_t _dtr);

    void update();


    tick_t getTime() { return currentTime; }

    bool shouldUpdate();
    bool shouldRender();

    tick_t getSleepTime();

    double getTimeScale() const;
    void setTimeScale(double value);

    // When there are very long updates for example when loading a level, the
    // gameloop tries to catch up, unless you call this method.
    void jumpToLive();
};

// use a global variable here so every object can access it easily
SAIGA_CORE_API extern GameTime gameTime;

}  // namespace Saiga
