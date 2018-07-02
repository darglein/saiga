/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/window/MainLoop.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/image/glImageFormat.h"
//#include "saiga/rendering/deferredRendering/deferred_renderer.h"
#include "saiga/rendering/renderer.h"
#include "saiga/rendering/program.h"

#include "saiga/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/framework.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/math.h"
#include "saiga/camera/camera.h"

#include <cstring>
#include <vector>
#include <ctime>
#include <thread>

namespace Saiga {

MainLoop::MainLoop(MainLoopInterface &renderer)
    : renderer(renderer), updating(renderer)
{

}

void MainLoop::updateUpdateGraph()
{
    ut = std::chrono::duration<double, std::milli>(updateTimer.getTime()).count();

    maxUpdateTime = std::max(ut,maxUpdateTime);

    avUt = 0;
    for(int i = 0 ;i < numGraphValues; ++i){
        avUt +=   imUpdateTimes[i];
    }
    avUt /= numGraphValues;

    imUpdateTimes[imCurrentIndexUpdate] = ut;
    imCurrentIndexUpdate = (imCurrentIndexUpdate+1) % numGraphValues;
}

void MainLoop::updateRenderGraph()
{

    ft = renderer.getTotalRenderTime();
    maxRenderTime = std::max(ft,maxRenderTime);

    avFt = 0;
    for(int i = 0 ;i < numGraphValues; ++i){
        avFt +=   imRenderTimes[i];
    }
    avFt /= numGraphValues;
    imRenderTimes[imCurrentIndexRender] = ft;
    imCurrentIndexRender = (imCurrentIndexRender+1) % numGraphValues;
}


void MainLoop::update(float dt)
{
    updateTimer.start();
    endParallelUpdate();
    updating.update(dt);
    startParallelUpdate(dt);
    updateTimer.stop();
    updateUpdateGraph();

    numUpdates++;

    upsTimer.stop();
    upsTimer.start();
}




void MainLoop::startParallelUpdate(float dt)
{

    if(parallelUpdate){
        semStartUpdate.notify();
    }else{
        parallelUpdateCaller(dt);
    }
}

void MainLoop::endParallelUpdate()
{
    if(parallelUpdate)
        semFinishUpdate.wait();
}

void MainLoop::parallelUpdateThread(float dt)
{
    semFinishUpdate.notify();
    semStartUpdate.wait();
    while(running){
        parallelUpdateCaller(dt);
        semFinishUpdate.notify();
        semStartUpdate.wait();
    }
}


void MainLoop::parallelUpdateCaller(float dt)
{
    updating.parallelUpdate(dt);
}

void MainLoop::render(float dt, float interpolation)
{
    interpolationTimer.start();
    //    renderer.renderer.interpolate(dt,interpolation);
    updating.interpolate(dt,interpolation);
    interpolationTimer.stop();

    renderCPUTimer.start();
    renderer.render();
    renderCPUTimer.stop();

    updateRenderGraph();
    numFrames++;

    swapBuffersTimer.start();
    //    if(windowParameters.finishBeforeSwap) glFinish();
    //    swapBuffers();
    renderer.swap();
    swapBuffersTimer.stop();

    fpsTimer.stop();
    fpsTimer.start();
}



void MainLoop::sleep(tick_t ticks)
{
    if(ticks > tick_t(0)){
        std::this_thread::sleep_for(ticks);
    }
}





void MainLoop::startMainLoop(MainLoopParameters params)
{
    parallelUpdate = params.parallelUpdate;
    printInfoMsg = params.printInfoMsg;
    gameTime.printInfoMsg = printInfoMsg;
    running = true;

    cout << "> Starting the main loop..." << endl;
    cout << "> updatesPerSecond=" << params.updatesPerSecond << " framesPerSecond=" << params.framesPerSecond << " maxFrameSkip=" << params.maxFrameSkip << endl;


    if(params.updatesPerSecond <= 0)
        params.updatesPerSecond = gameTime.base.count();
    if(params.framesPerSecond <= 0)
        params.framesPerSecond = gameTime.base.count();


    float updateDT = 1.0f / params.updatesPerSecond;
    targetUps = params.updatesPerSecond;
    //    float framesDT = 1.0f / framesPerSecond;

    tick_t ticksPerUpdate = gameTime.base / params.updatesPerSecond;
    tick_t ticksPerFrame = gameTime.base / params.framesPerSecond;

    tick_t ticksPerInfo = std::chrono::duration_cast<tick_t>(gameTime.base * params.mainLoopInfoTime);

    tick_t ticksPerScreenshot = std::chrono::duration_cast<tick_t>(gameTime.base * 5.0f);

    //    if(windowParameters.debugScreenshotTime < 0)
    //        ticksPerScreenshot = std::chrono::duration_cast<tick_t>(std::chrono::hours(100000));

    gameTime.init(ticksPerUpdate,ticksPerFrame);


    tick_t nextInfoTick = gameTime.getTime();
    tick_t nextScreenshotTick = gameTime.getTime() + ticksPerScreenshot;

    if(!params.catchUp){
        gameTime.maxGameLoopDelay = std::chrono::duration_cast<tick_t>(std::chrono::milliseconds(1));
    }


    if(parallelUpdate){
        updateThread = std::thread(&MainLoop::parallelUpdateThread,this,updateDT);
    }


    while(true)
    {
        if(updating.shouldClose()){
            break;
        }

        //With this loop we are able to skip frames if the system can't keep up.
        for(int i = 0; i <= params.maxFrameSkip && gameTime.shouldUpdate(); ++i){
            update(updateDT);
        }

        if(gameTime.shouldRender()){
            render(updateDT,gameTime.interpolation);
        }

        if(printInfoMsg && gameTime.getTime() > nextInfoTick){
            auto gt = std::chrono::duration_cast<std::chrono::seconds>(gameTime.getTime());
            cout << "> Time: " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/" << numFrames << "  UPS/FPS: " << (1000.0f/upsTimer.getTimeMS()) << "/" << (1000.0f/fpsTimer.getTimeMS()) << endl;
            nextInfoTick += ticksPerInfo;
        }

        //        if(gameTime.getTime() > nextScreenshotTick){
        //            SAIGA_ASSERT(0);
        //            string file = windowParameters.debugScreenshotPath+getTimeString()+".png";
        //            this->screenshot(file);
        //            nextScreenshotTick += ticksPerScreenshot;
        //        }

        //sleep until the next interesting event
        sleep(gameTime.getSleepTime());
        assert_no_glerror_end_frame();
    }
    running = false;

    if(parallelUpdate){
        //cleanup the update thread
        cout << "Finished main loop. Exiting update thread." << endl;
        endParallelUpdate();
        semStartUpdate.notify();
        updateThread.join();
    }

    auto gt = std::chrono::duration_cast<std::chrono::seconds>(gameTime.getTime());
    cout << "> Main loop finished in " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/" << numFrames  << endl;
}

}
