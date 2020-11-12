/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/rendering/lighting/box_light.h"
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"

namespace Saiga
{
ForwardLighting::ForwardLighting() : RendererLighting()
{
    pointLightBuffer.createGLBuffer(nullptr, sizeof(LightData), GL_DYNAMIC_DRAW);
}

ForwardLighting::~ForwardLighting() {}

void ForwardLighting::initRender()
{
    RendererLighting::initRender();
    pointLightBuffer.bind(LIGHT_DATA_BINDING_POINT);
    LightData ld;
    ld.plCount = 0;
    for (auto pl : pointLights)
    {
        if (!pl->shouldRender()) continue;
        ld.plPositions[ld.plCount]    = make_vec4(pl->getPosition(), 0.0f);
        ld.plColors[ld.plCount]       = make_vec4(pl->getColorDiffuse(), pl->getIntensity());
        ld.plAttenuations[ld.plCount] = make_vec4(pl->getAttenuation(), pl->getRadius());
        ld.plCount++;
        if (ld.plCount >= MAX_PL_COUNT) break;  // just ignore too many lights...
    }
    pointLightBuffer.updateBuffer(&ld, sizeof(LightData), 0);
    plCount       = ld.plCount;
    visibleLights = plCount;
}

void ForwardLighting::render(Camera* cam, const ViewPort& viewPort)
{
    // Does nothing
    RendererLighting::render(cam, viewPort);

    if (drawDebug)
    {
        //        glDepthMask(GL_TRUE);
        renderDebug(cam);
        //        glDepthMask(GL_FALSE);
    }
    assert_no_glerror();
}

void ForwardLighting::renderImGui(bool* p_open)
{
    RendererLighting::renderImGui();
    if (!ImGui::Begin("ForwardLighting")) return;
    ImGui::Text("Rendered Point Lights: %d", plCount);
    if (ImGui::Button("Clear Point Lights"))
    {
        pointLights.clear();
    }

    int32_t count = pointLights.size();
    if (ImGui::InputInt("Point Light Count (wanted)", &count))
    {
        if (count > pointLights.size())
        {
            count -= pointLights.size();
            for (int32_t i = 0; i < count; ++i)
            {
                std::shared_ptr<PointLight> light = std::make_shared<PointLight>();
                light->setAttenuation(AttenuationPresets::Quadratic);
                light->setIntensity(1);


                light->setRadius(linearRand(2, 8));

                light->setPosition(linearRand(vec3(-16, 0.5, -16), vec3(16, 2, 16)));

                light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
                light->calculateModel();

                light->createShadowMap(512, 512, ShadowQuality::HIGH);
                light->enableShadows();
                AddLight(light);
            }
        }
        else if (count < pointLights.size())
        {
            count = pointLights.size() - count;
            for (int32_t i = 0; i < count; ++i)
            {
                pointLights.erase(--pointLights.end());
            }
        }
    }
    if (ImGui::Button("Normalize Point Lights"))
    {
        for (auto pl : pointLights)
        {
            float intensity = pl->getIntensity();
            intensity       = 1.0f / pl->getRadius();
            pl->setIntensity(intensity);
        }
    }
    ImGui::End();
}

}  // namespace Saiga