/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include <saiga/util/semaphore.h>
#include "saiga/time/gameTime.h"
#include "saiga/geometry/ray.h"
#include "saiga/imgui/imgui_renderer.h"
#include "saiga/rendering/renderer.h"
#include "saiga/window/windowParameters.h"
#include <thread>

namespace Saiga {


struct MainLoopInterface
{
    virtual void render() = 0;
    virtual void swap() = 0;
    virtual float getTotalRenderTime() { return 0; }
    virtual bool shouldClose() { return false; }
    virtual void update(float dt) {}
    virtual void parallelUpdate(float dt) { (void)dt; }
    virtual void interpolate(float dt, float alpha) {}
};



class SAIGA_GLOBAL MainLoop
{
public:
    MainLoopInterface& renderer;
    MainLoopInterface& updating;

    MainLoop(MainLoopInterface& renderer);
    void startMainLoop(MainLoopParameters params = MainLoopParameters());

private:

    //total number of updateticks/frames rendered so far
    int numUpdates = 0;
    int numFrames = 0;

    //game loop running
    bool running = false;

    //basic variables for the parallel update
    Semaphore semStartUpdate, semFinishUpdate;
    std::thread updateThread;
    bool parallelUpdate = false;


    tick_t gameLoopDelay = tick_t(0);

    bool gameloopDropAccumulatedUpdates = false;
    bool printInfoMsg = true;

    //for imgui graph
    bool showImgui = true;
    static const int numGraphValues = 80;
    float ut=0, ft=0;
    float avFt = 0, avUt;
    int imCurrentIndexUpdate = 0;
    int imCurrentIndexRender = 0;
    float imUpdateTimes[numGraphValues];
    float imRenderTimes[numGraphValues];
    bool showImguiDemo = false;
    float maxUpdateTime = 1;
    float maxRenderTime = 1;
    int targetUps = 60;


    ExponentialTimer updateTimer, interpolationTimer, renderCPUTimer, swapBuffersTimer;
    AverageTimer fpsTimer, upsTimer;

    void updateRenderGraph();
    void updateUpdateGraph();


    void update(float dt);
    void render(float dt, float interpolation);
    void startParallelUpdate(float dt);
    void parallelUpdateCaller(float dt);
    void endParallelUpdate();
    void parallelUpdateThread(float dt);


    void sleep(tick_t ticks);

};

}
