/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"

namespace Saiga
{
ForwardLighting::ForwardLighting() : clearColor(make_vec4(0.0)), lightCount(0), debugDraw(false), width(0), height(0)
{
    pointLightBuffer.createGLBuffer(nullptr, sizeof(LightData), GL_DYNAMIC_DRAW);
}

ForwardLighting::~ForwardLighting() {}

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

void ForwardLighting::addPointLight(std::shared_ptr<PointLight> light)
{
    pointLights.push_back(light);
    lightCount++;
}

void ForwardLighting::removePointLight(std::shared_ptr<PointLight> light)
{
    pointLights.erase(std::find(pointLights.begin(), pointLights.end(), light));
    lightCount--;
}

void ForwardLighting::render(RenderingInterface* renderingInterface, Camera* camera)
{
    // mode: no shader loop
    pointLightBuffer.bind(LIGHT_DATA_BINDING_POINT);
    LightData ld;
    ld.plCount = 0;
    // static float delta = 0.0f;
    for (auto pl : pointLights)
    {
        // auto p = pl->getPosition();
        // p.x() += (sin(delta + pow(p.y() * pl->getRadius(), 2.0f)) * pl->getRadius() * 0.01f);
        // p.z() += (cos(delta + pow(p.y() * pl->getRadius(), 3.0f)) * pl->getRadius() * 0.01f);
        // pl->setPosition(p);
        ld.plPositions[ld.plCount]    = make_vec4(pl->getPosition(), 0.0f);
        ld.plColors[ld.plCount]       = make_vec4(pl->getColorDiffuse(), pl->getIntensity());
        ld.plAttenuations[ld.plCount] = make_vec4(pl->getAttenuation(), pl->getRadius());
        ld.plCount++;
        if (ld.plCount >= MAX_PL_COUNT) break;  // just ignore two many lights...
    }
    pointLightBuffer.updateBuffer(&ld, sizeof(LightData), 0);
    renderingInterface->render(camera, RenderPass::Forward);
    // delta += 0.01f;
}

void ForwardLighting::printTimings() {}

void ForwardLighting::renderImGui(bool* p_open)
{
    if (ImGui::Button("Clear Lights"))
    {
        pointLights.clear();
    }

    int32_t count = pointLights.size();
    if (ImGui::InputInt("Point Light Count", &count))
    {
        count = count > MAX_PL_COUNT ? MAX_PL_COUNT : count;
        if (count > pointLights.size())
        {
            count -= pointLights.size();
            for (int32_t i = 0; i < count; ++i)
            {
                std::shared_ptr<PointLight> pl = std::make_shared<PointLight>();
                pl->setAttenuation(vec3(1.0f, 1.0f, 1.0f));
                pl->setRadius(Random::sampleDouble(2.0f, 6.0f));
                pl->setPosition(vec3(Random::sampleDouble(-16.0f, 16.0f),
                                     Random::sampleDouble(1.0f, pl->getRadius() * 0.5f),
                                     Random::sampleDouble(-16.0f, 16.0f)));
                pl->setColorDiffuse(vec3(Random::sampleDouble(0.0f, 1.0f), Random::sampleDouble(0.0f, 1.0f),
                                         Random::sampleDouble(0.0f, 1.0f)));
                pl->setIntensity(Random::sampleDouble(0.5f, 2.0f));
                addPointLight(pl);
            }
        }
        else if (count < pointLights.size())
        {
            count = pointLights.size() - count;
            for (int32_t i = 0; i < count; ++i)
            {
                pointLights.pop_back();
            }
        }
    }
    if (ImGui::Button("Normalize Lights"))
    {
        for (auto pl : pointLights)
        {
            float intensity = pl->getIntensity();
            intensity       = 1.0f / pl->getRadius();
            pl->setIntensity(intensity);
        }
    }
}

}  // namespace Saiga