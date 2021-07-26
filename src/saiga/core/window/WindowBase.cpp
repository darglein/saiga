/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "WindowBase.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"

#include "internal/noGraphicsAPI.h"

#include <iostream>
namespace Saiga
{
WindowBase::WindowBase(WindowParameters _windowParameters) : mainLoop(*this), windowParameters(_windowParameters) {}

WindowBase::~WindowBase() {}

void WindowBase::close()
{
    running = false;
}


void WindowBase::resize(int width, int height)
{
    this->windowParameters.width  = width;
    this->windowParameters.height = height;
    // renderer->resize(width, height);
}


std::string WindowBase::getTimeString()
{
    time_t t       = time(0);  // get time now
    struct tm* now = localtime(&t);

    std::string str;
    str = std::to_string(now->tm_year + 1900) + '-' + std::to_string(now->tm_mon + 1) + '-' +
          std::to_string(now->tm_mday) + '_' + std::to_string(now->tm_hour) + '-' + std::to_string(now->tm_min) + '-' +
          std::to_string(now->tm_sec);

    ;

    return str;
}

void WindowBase::interpolate(float dt, float alpha)
{
    SAIGA_ASSERT(updating);
    updating->interpolate(dt, alpha);
}


void WindowBase::render()
{
    //    SAIGA_ASSERT(currentCamera);
    SAIGA_ASSERT(renderer);

    RenderInfo renderInfo;
    renderInfo.camera = activeCameras.front().first;


    if (renderer) renderer->render(renderInfo);
}


void WindowBase::startMainLoop(MainLoopParameters params)
{
    //    SAIGA_ASSERT(currentCamera);
    running = true;
    mainLoop.startMainLoop(params);
}



}  // namespace Saiga
