/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

namespace Saiga
{

ForwardLighting::ForwardLighting()
    : clearColor(make_vec4(0.0))
    , lightCount(0)
    , visibleLights(0)
    , debugDraw(false)
    , width(0)
    , height(0)
{
}

ForwardLighting::~ForwardLighting()
{
}

void ForwardLighting::init(int _width, int _height)
{
    this->width  = _width;
    this->height = _height;
}

void ForwardLighting::resize(int _width, int _height)
{
    this->width  = _width;
    this->height = _height;
}

std::shared_ptr<StandardPointLight> ForwardLighting::addPointLight(StandardPointLightParameters params)
{
    std::shared_ptr<StandardPointLight> light = std::make_shared<StandardPointLight>(params);
    pointLights.push_back(light);
    lightCount++;
    return light;
}

void ForwardLighting::removePointLight(std::shared_ptr<StandardPointLight> light)
{
    pointLights.erase(std::find(pointLights.begin(), pointLights.end(), light));
    lightCount--;
}

void ForwardLighting::initRender()
{
}

void ForwardLighting::endRender()
{
}

void ForwardLighting::printTimings()
{

}

void ForwardLighting::renderImGui(bool* p_open)
{

}

} // namespace Saiga