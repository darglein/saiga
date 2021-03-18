/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"

namespace Saiga
{
using namespace uber;

ForwardLighting::ForwardLighting(GLTimerSystem* timer) : RendererLighting(timer)
{
    int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

    maximumNumberOfDirectionalLights =
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLight::ShaderData));
    maximumNumberOfPointLights =
        std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLight::ShaderData));
    maximumNumberOfSpotLights = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLight::ShaderData));

    lightDataBufferDirectional.createGLBuffer(
        nullptr, sizeof(DirectionalLight::ShaderData) * maximumNumberOfDirectionalLights, GL_DYNAMIC_DRAW);
    lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLight::ShaderData) * maximumNumberOfPointLights,
                                        GL_DYNAMIC_DRAW);
    lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLight::ShaderData) * maximumNumberOfSpotLights,
                                       GL_DYNAMIC_DRAW);
    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);
}

ForwardLighting::~ForwardLighting() {}

void ForwardLighting::initRender()
{
    RendererLighting::initRender();
    lightDataBufferPoint.bind(POINT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferSpot.bind(SPOT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferDirectional.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
    LightInfo li;
    LightData ld;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.directionalLightCount = 0;
    li.clusterEnabled        = false;

    // Point Lights
    for (auto pl : pointLights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;
        ld.pointLights.push_back(pl->GetShaderData());

        li.pointLightCount++;
    }

    // Spot Lights
    for (auto sl : spotLights)
    {
        if (li.spotLightCount >= maximumNumberOfSpotLights) break;  // just ignore too many lights...
        if (!sl->shouldRender()) continue;
        ld.spotLights.push_back(sl->GetShaderData());

        li.spotLightCount++;
    }

    // Directional Lights
    for (auto dl : directionalLights)
    {
        if (li.directionalLightCount >= maximumNumberOfDirectionalLights) break;  // just ignore too many lights...
        if (!dl->shouldRender()) continue;
        ld.directionalLights.push_back(dl->GetShaderData());

        li.directionalLightCount++;
    }

    lightDataBufferPoint.updateBuffer(ld.pointLights.data(), sizeof(PointLight::ShaderData) * li.pointLightCount, 0);
    lightDataBufferSpot.updateBuffer(ld.spotLights.data(), sizeof(SpotLight::ShaderData) * li.spotLightCount, 0);
    lightDataBufferDirectional.updateBuffer(ld.directionalLights.data(),
                                            sizeof(DirectionalLight::ShaderData) * li.directionalLightCount, 0);

    lightInfoBuffer.updateBuffer(&li, sizeof(LightInfo), 0);
    visibleLights = li.pointLightCount + li.spotLightCount + li.directionalLightCount;
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

void ForwardLighting::setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights)
{
    maxDirectionalLights = std::max(0, maxDirectionalLights);
    maxPointLights       = std::max(0, maxPointLights);
    maxSpotLights        = std::max(0, maxSpotLights);

    int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

    maximumNumberOfDirectionalLights =
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLight::ShaderData));
    maximumNumberOfPointLights =
        std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLight::ShaderData));
    maximumNumberOfSpotLights = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLight::ShaderData));


    if (maximumNumberOfDirectionalLights != maxDirectionalLights)
    {
        lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLight::ShaderData) * maxDirectionalLights,
                                                  GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfPointLights != maxPointLights)
    {
        lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLight::ShaderData) * maxPointLights, GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfSpotLights != maxSpotLights)
    {
        lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLight::ShaderData) * maxSpotLights, GL_DYNAMIC_DRAW);
    }

    maximumNumberOfDirectionalLights = maxDirectionalLights;
    maximumNumberOfPointLights       = maxPointLights;
    maximumNumberOfSpotLights        = maxSpotLights;
}


void ForwardLighting::renderImGui()
{
    RendererLighting::renderImGui();
}

}  // namespace Saiga
