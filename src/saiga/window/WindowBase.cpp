/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/window/WindowBase.h"
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



WindowBase::WindowBase(WindowParameters _windowParameters)
    :windowParameters(_windowParameters), mainLoop(*this)
{
}

WindowBase::~WindowBase(){
    //    delete renderer;
}

void WindowBase::close(){
    cout<<"Window: close"<<endl;
    running = false;
}




void WindowBase::resize(int width, int height)
{
    this->windowParameters.width = width;
    this->windowParameters.height = height;
    renderer->resize(width,height);
}


std::string WindowBase::getTimeString()
{
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );

    std::string str;
    str =     std::to_string(now->tm_year + 1900) + '-'
            + std::to_string(now->tm_mon + 1) + '-'
            + std::to_string(now->tm_mday) + '_'
            + std::to_string(now->tm_hour) + '-'
            + std::to_string(now->tm_min) + '-'
            + std::to_string(now->tm_sec);

    ;

    return str;
}


Ray WindowBase::createPixelRay(const vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/getWidth()-1.f,1.f-(2*pixel.y/getHeight()),0,1.f);
    p = glm::inverse(WindowBase::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(WindowBase::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

Ray WindowBase::createPixelRay(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(WindowBase::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

vec3 WindowBase::screenToWorld(const vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/getWidth()-1.f,1.f-(2*pixel.y/getHeight()),0,1.f);
    p = glm::inverse(WindowBase::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(WindowBase::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}


vec3 WindowBase::screenToWorld(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(WindowBase::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}



vec2 WindowBase::projectToScreen(const vec3 &pos) const
{
    vec4 r = WindowBase::currentCamera->proj * WindowBase::currentCamera->view * vec4(pos,1);
    r /= r.w;

    vec2 pixel;
    pixel.x = (r.x +1.f)*getWidth() *0.5f;
    pixel.y = -(r.y - 1.f) * getHeight() * 0.5f;

    return pixel;
}


void WindowBase::interpolate(float dt, float alpha)
{
    updating->interpolate(dt,alpha);
}


void WindowBase::render()
{
    SAIGA_ASSERT(currentCamera);
    if(renderer)
        renderer->render(currentCamera);

}


void WindowBase::startMainLoop(MainLoopParameters params)
{
    SAIGA_ASSERT(currentCamera);
    running = true;
    mainLoop.startMainLoop(params);
}

}
