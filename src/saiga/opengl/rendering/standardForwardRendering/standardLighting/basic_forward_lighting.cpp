/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "basic_forward_lighting.h"

namespace Saiga
{

BasicForwardLighting::BasicForwardLighting()
    : clearColor(make_vec4(0.0))
    , lightCount(0)
    , visibleLights(0)
    , debugDraw(false)
    , width(0)
    , height(0)
{
}

BasicForwardLighting::~BasicForwardLighting()
{ 
}

void BasicForwardLighting::init(int _width, int _height)
{
    this->width  = _width;
    this->height = _height;
}

void BasicForwardLighting::resize(int _width, int _height)
{
    this->width  = _width;
    this->height = _height;
}

std::shared_ptr<StandardPointLight> BasicForwardLighting::addPointLight(StandardPointLightParameters params)
{
    std::shared_ptr<StandardPointLight> light = std::make_shared<StandardPointLight>(params);
    pointLights.push_back(light);
    lightCount++;
    return light;
}

void BasicForwardLighting::removePointLight(std::shared_ptr<StandardPointLight> light)
{
    pointLights.erase(std::find(pointLights.begin(), pointLights.end(), light));
    lightCount--;
}

void BasicForwardLighting::initRender()
{
}

void BasicForwardLighting::endRender()
{
}

void BasicForwardLighting::printTimings()
{
    
}

void BasicForwardLighting::renderImGui(bool* p_open)
{
    
}

} // namespace Saiga