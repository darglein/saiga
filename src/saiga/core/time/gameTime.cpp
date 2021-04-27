/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "gameTime.h"

#include "saiga/core/math/math.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
GameTime gameTime;

void GameTime::init(tick_t _dt, tick_t _dtr)
{
    maxGameLoopDelay = std::chrono::duration_cast<tick_t>(std::chrono::hours(1000));
    dt               = _dt;
    dtr              = _dtr;

    gameTimer.start();

    realTime     = gameTimer.stop();
    lastRealTime = realTime;

    update();
    nextUpdateTime = scaledTime;
}


double GameTime::getTimeScale() const
{
    return timeScale;
}

void GameTime::setTimeScale(double value)
{
    timeScale = value;
}

void GameTime::jumpToLive()
{
    jtl = true;
}


void GameTime::update()
{
    realTime        = gameTimer.stop();
    auto step       = realTime - lastRealTime;
    auto scaledStep = std::chrono::duration_cast<tick_t>(step * timeScale);
    scaledTime += scaledStep;
    lastRealTime = realTime;
}

bool GameTime::shouldUpdate()
{
    update();
    auto currentDelay = scaledTime - nextUpdateTime;

    if (currentDelay > maxGameLoopDelay)
    {
        jtl = true;
    }


    if (jtl)
    {
        if (currentDelay.count() > 0)
        {
            if (printInfoMsg)
            {
                //                std::cout << "> Advancing game time to live. Adding a delay of " <<
                //                std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(currentDelay).count()
                //                << " ms" << std::endl;
            }
            scaledTime    = nextUpdateTime;
            nextFrameTime = realTime;
        }
        jtl = false;
    }



    if (currentDelay > tick_t(0))
    {
        actualUpdateTime = scaledTime;
        updatetime       = nextUpdateTime;
        currentTime      = updatetime;
        nextUpdateTime += dt;
        return true;
    }
    else
    {
        return false;
    }
}

bool GameTime::shouldRender()
{
    update();
    if (realTime > nextFrameTime)
    {
        //        updatetime = nextFrameTick;

        tick_t ticksSinceLastUpdate = scaledTime - actualUpdateTime;

        renderTime  = updatetime + ticksSinceLastUpdate;
        currentTime = renderTime;

        //        calculate the interpolation value. Useful when the framerate is higher than the update rate
        interpolation = (double)ticksSinceLastUpdate.count() / (nextUpdateTime - updatetime).count();
        interpolation = clamp(interpolation, 0.0, 1.0);


        nextFrameTime += dtr;
        return true;
    }
    else
    {
        return false;
    }
}

tick_t GameTime::getSleepTime()
{
    update();

    auto timeTillU = nextUpdateTime - scaledTime;
    auto timeTillR = nextFrameTime - realTime;

    tick_t nextEvent = timeTillU < timeTillR ? timeTillU : timeTillR;
    return nextEvent;
}

}  // namespace Saiga
