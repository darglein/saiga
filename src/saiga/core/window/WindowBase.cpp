/**
 * Copyright (c) 2017 Darius Rückert
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
WindowBase::WindowBase(WindowParameters _windowParameters) : mainLoop(*this), windowParameters(_windowParameters)
{
    showRendererImgui = windowParameters.showRendererWindow;
}

WindowBase::~WindowBase() {}

void WindowBase::close()
{
    std::cout << "Window: close" << std::endl;
    running = false;
}


void WindowBase::resize(int width, int height)
{
    this->windowParameters.width  = width;
    this->windowParameters.height = height;
    renderer->resize(width, height);
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


Ray WindowBase::createPixelRay(const vec2& pixel) const
{
    vec4 p = vec4(2 * pixel[0] / getWidth() - 1.f, 1.f - (2 * pixel[1] / getHeight()), 0, 1.f);
    p      = inverse(WindowBase::getCamera()->proj) * p;
    p /= p[3];

    mat4 inverseView = inverse(WindowBase::getCamera()->view);
    vec3 ray_world   = make_vec3(inverseView * p);
    vec3 origin      = make_vec3(inverseView.col(3));
    return Ray(normalize(vec3(ray_world - origin)), origin);
}

Ray WindowBase::createPixelRay(const vec2& pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2 * pixel[0] / resolution[0] - 1.f, 1.f - (2 * pixel[1] / resolution[1]), 0, 1.f);
    p      = inverseProj * p;
    p /= p[3];

    mat4 inverseView = inverse(WindowBase::getCamera()->view);
    vec3 ray_world   = make_vec3(inverseView * p);
    vec3 origin      = make_vec3(inverseView.col(3));
    return Ray(normalize(vec3(ray_world - origin)), origin);
}

vec3 WindowBase::screenToWorld(const vec2& pixel) const
{
    vec4 p = vec4(2 * pixel[0] / getWidth() - 1.f, 1.f - (2 * pixel[1] / getHeight()), 0, 1.f);
    p      = inverse(WindowBase::getCamera()->proj) * p;
    p /= p[3];

    mat4 inverseView = inverse(WindowBase::getCamera()->view);
    vec3 ray_world   = make_vec3(inverseView * p);
    return ray_world;
}


vec3 WindowBase::screenToWorld(const vec2& pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2 * pixel[0] / resolution[0] - 1.f, 1.f - (2 * pixel[1] / resolution[1]), 0, 1.f);
    p      = inverseProj * p;
    p /= p[3];

    mat4 inverseView = inverse(WindowBase::getCamera()->view);
    vec3 ray_world   = make_vec3(inverseView * p);
    return ray_world;
}



vec2 WindowBase::projectToScreen(const vec3& pos) const
{
    vec4 r = WindowBase::getCamera()->proj * WindowBase::getCamera()->view * make_vec4(pos, 1);
    r /= r[3];

    vec2 pixel;
    pixel[0] = (r[0] + 1.f) * getWidth() * 0.5f;
    pixel[1] = -(r[1] - 1.f) * getHeight() * 0.5f;

    return pixel;
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
    renderInfo.cameras = activeCameras;
    if (renderer) renderer->render(renderInfo);
}


void WindowBase::startMainLoop(MainLoopParameters params)
{
    //    SAIGA_ASSERT(currentCamera);
    running = true;
    mainLoop.startMainLoop(params);
}



}  // namespace Saiga
