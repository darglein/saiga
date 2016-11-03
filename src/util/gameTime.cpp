#include "saiga/util/gameTime.h"
#include "saiga/util/glm.h"
#include <iostream>
GameTime gameTime;

void GameTime::init(tick_t _dt, tick_t _dtr)
{
    dt = _dt;
    dtr = _dtr;

    gameTimer.start();

    realTime = gameTimer.stop();
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

    update();

    auto delay = scaledTime - nextUpdateTime;
    std::cout << "> Advancing game time to live. Adding a delay of " << std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(delay).count() << " ms" << std::endl;

    scaledTime = nextUpdateTime;
    nextFrameTime = realTime;
}


void GameTime::update()
{
    realTime = gameTimer.stop();
    auto step = realTime - lastRealTime;
    auto scaledStep = std::chrono::duration_cast<tick_t>(step * timeScale);
    scaledTime += scaledStep;
    lastRealTime = realTime;
}

bool GameTime::shouldUpdate()
{
    update();
    if(scaledTime > nextUpdateTime){
        actualUpdateTime = scaledTime;
        updatetime = nextUpdateTime;
        currentTime = updatetime;
        nextUpdateTime += dt;
        return true;
    }else{
        return false;
    }
}

bool GameTime::shouldRender()
{
    update();
    if(realTime > nextFrameTime){
        //        updatetime = nextFrameTick;

        tick_t ticksSinceLastUpdate = scaledTime - actualUpdateTime;

        renderTime = updatetime + ticksSinceLastUpdate;
        currentTime = renderTime;

        //        calculate the interpolation value. Useful when the framerate is higher than the update rate
        interpolation = (double)ticksSinceLastUpdate.count() / (nextUpdateTime - updatetime).count();
        interpolation = glm::clamp(interpolation,0.0,1.0);


        nextFrameTime += dtr;
        return true;
    }else{
        return false;
    }
}

tick_t GameTime::getSleepTime()
{
    update();
    tick_t nextEvent = nextFrameTime < nextUpdateTime? nextFrameTime: nextUpdateTime;
    return nextEvent - scaledTime;
}
