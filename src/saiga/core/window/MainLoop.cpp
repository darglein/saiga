/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "MainLoop.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"


#include <iostream>

namespace Saiga
{
void MainLoopParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    updatesPerSecond = ini.GetAddLong("mainloop", "updatesPerSecond", updatesPerSecond);
    framesPerSecond  = ini.GetAddLong("mainloop", "framesPerSecond", framesPerSecond);
    mainLoopInfoTime = ini.GetAddDouble("mainloop", "mainLoopInfoTime", mainLoopInfoTime);
    maxFrameSkip     = ini.GetAddLong("mainloop", "maxFrameSkip", maxFrameSkip);
    parallelUpdate   = ini.GetAddBool("mainloop", "parallelUpdate", parallelUpdate);
    catchUp          = ini.GetAddBool("mainloop", "catchUp", catchUp);
    printInfoMsg     = ini.GetAddBool("mainloop", "printInfoMsg", printInfoMsg);

    if (ini.changed()) ini.SaveFile(file.c_str());
}

MainLoop::MainLoop(MainLoopInterface& renderer)
    : renderer(renderer),
      updating(renderer),
      updateTimer(0.97f),
      interpolationTimer(0.97f),
      renderCPUTimer(0.97f),
      swapBuffersTimer(0.97f),
      fpsTimer(50),
      upsTimer(50),
      updateGraph("Update", 80),
      renderGraph("Render", 80)
{
}


void MainLoop::update(float dt)
{
    updateTimer.start();
    endParallelUpdate();
    updating.update(dt);
    startParallelUpdate(dt);
    updateTimer.stop();
    updateGraph.addTime(
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(updateTimer.getTime()).count());

    numUpdates++;

    upsTimer.stop();
    upsTimer.start();
}

void MainLoop::startParallelUpdate(float dt)
{
    if (parallelUpdate)
    {
        semStartUpdate.notify();
    }
    else
    {
        parallelUpdateCaller(dt);
    }
}

void MainLoop::endParallelUpdate()
{
    if (parallelUpdate) semFinishUpdate.wait();
}

void MainLoop::parallelUpdateThread(float dt)
{
    semFinishUpdate.notify();
    semStartUpdate.wait();
    while (running)
    {
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
    updating.interpolate(dt, interpolation);
    interpolationTimer.stop();

    renderCPUTimer.start();
    renderer.render();
    renderCPUTimer.stop();


    renderGraph.addTime(renderer.getTotalRenderTime());
    numFrames++;

    swapBuffersTimer.start();
    renderer.swap();
    swapBuffersTimer.stop();

    fpsTimer.stop();
    fpsTimer.start();
}



void MainLoop::sleep(tick_t ticks)
{
    if (ticks > tick_t(0))
    {
        std::this_thread::sleep_for(ticks);
    }
}



void MainLoop::startMainLoop(MainLoopParameters params)
{
    parallelUpdate        = params.parallelUpdate;
    printInfoMsg          = params.printInfoMsg;
    gameTime.printInfoMsg = printInfoMsg;
    running               = true;

    VLOG(1) << "> Starting the main loop...";
    VLOG(1) << "> updatesPerSecond=" << params.updatesPerSecond << " framesPerSecond=" << params.framesPerSecond
            << " maxFrameSkip=" << params.maxFrameSkip;


    if (params.updatesPerSecond <= 0) params.updatesPerSecond = gameTime.base.count();
    if (params.framesPerSecond <= 0) params.framesPerSecond = gameTime.base.count();


    float updateDT = 1.0f / params.updatesPerSecond;
    targetUps      = params.updatesPerSecond;
    //    float framesDT = 1.0f / framesPerSecond;

    tick_t ticksPerUpdate = gameTime.base / params.updatesPerSecond;
    tick_t ticksPerFrame  = gameTime.base / params.framesPerSecond;

    tick_t ticksPerInfo = std::chrono::duration_cast<tick_t>(gameTime.base * params.mainLoopInfoTime);

    //    tick_t ticksPerScreenshot = std::chrono::duration_cast<tick_t>(gameTime.base * 5.0f);

    //    if(windowParameters.debugScreenshotTime < 0)
    //        ticksPerScreenshot = std::chrono::duration_cast<tick_t>(std::chrono::hours(100000));

    gameTime.init(ticksPerUpdate, ticksPerFrame);


    tick_t nextInfoTick = gameTime.getTime();
    //    tick_t nextScreenshotTick = gameTime.getTime() + ticksPerScreenshot;

    if (!params.catchUp)
    {
        gameTime.maxGameLoopDelay = std::chrono::duration_cast<tick_t>(std::chrono::milliseconds(1));
    }


    if (parallelUpdate)
    {
        updateThread = std::thread(&MainLoop::parallelUpdateThread, this, updateDT);
    }


    while (true)
    {
        if (updating.shouldClose())
        {
            break;
        }

        // With this loop we are able to skip frames if the system can't keep up.
        for (int i = 0; i <= params.maxFrameSkip && gameTime.shouldUpdate(); ++i)
        {
            update(updateDT);
        }

        if (gameTime.shouldRender())
        {
            render(updateDT, gameTime.interpolation);
        }

        if (printInfoMsg && gameTime.getTime() > nextInfoTick)
        {
            auto gt = std::chrono::duration_cast<std::chrono::seconds>(gameTime.getTime());
            std::cout << "> Time: " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/"
                      << numFrames << "  UPS/FPS: " << (1000.0f / upsTimer.getTimeMS()) << "/"
                      << (1000.0f / fpsTimer.getTimeMS()) << std::endl;
            nextInfoTick += ticksPerInfo;
        }

        //        if(gameTime.getTime() > nextScreenshotTick){
        //            SAIGA_ASSERT(0);
        //            string file = windowParameters.debugScreenshotPath+getTimeString()+".png";
        //            this->screenshot(file);
        //            nextScreenshotTick += ticksPerScreenshot;
        //        }

        // sleep until the next interesting event
        sleep(gameTime.getSleepTime());
        //        assert_no_glerror_end_frame();
    }
    running = false;

    if (parallelUpdate)
    {
        // cleanup the update thread
        std::cout << "Finished main loop. Exiting update thread." << std::endl;
        endParallelUpdate();
        semStartUpdate.notify();
        updateThread.join();
    }

    auto gt = std::chrono::duration_cast<std::chrono::seconds>(gameTime.getTime());
    VLOG(1) << "> Main loop finished in " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/"
            << numFrames;
}

void MainLoop::renderImGuiInline()
{
    updateGraph.renderImGui();
    renderGraph.renderImGui();

    ImGui::Text("Swap Time: %fms", swapBuffersTimer.getTimeMS());
    ImGui::Text("Interpolate Time: %fms", interpolationTimer.getTimeMS());
    ImGui::Text("Render CPU Time: %fms", renderCPUTimer.getTimeMS());

    ImGui::Text("Running: %d", running);
    ImGui::Text("numUpdates: %d", numUpdates);
    ImGui::Text("numFrames: %d", numFrames);

    std::chrono::duration<double, std::milli> dt = gameTime.dt;
    ImGui::Text("Timestep: %fms", dt.count());

    std::chrono::duration<double, std::milli> delay = gameLoopDelay;
    ImGui::Text("Delay: %fms", delay.count());

    float scale = gameTime.getTimeScale();
    ImGui::SliderFloat("Time Scale", &scale, 0, 5);
    gameTime.setTimeScale(scale);

    //    ImGui::Checkbox("showImguiDemo",&showImguiDemo);
    //    if(mainLoop.showImguiDemo){
    //        ImGui::SetNextWindowPos(ImVec2(340, 0), ImGuiCond_FirstUseEver);
    //        ImGui::ShowTestWindow(&showImguiDemo);
    //    }
}

}  // namespace Saiga
