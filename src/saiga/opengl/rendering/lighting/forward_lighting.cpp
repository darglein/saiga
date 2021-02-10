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

ForwardLighting::ForwardLighting() : RendererLighting()
{
    int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

    maximumNumberOfDirectionalLights =
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLightData));
    maximumNumberOfPointLights = std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLightData));
    maximumNumberOfSpotLights  = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLightData));

    lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLightData) * maximumNumberOfDirectionalLights,
                                              GL_DYNAMIC_DRAW);
    lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLightData) * maximumNumberOfPointLights, GL_DYNAMIC_DRAW);
    lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLightData) * maximumNumberOfSpotLights, GL_DYNAMIC_DRAW);
    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);
}

ForwardLighting::~ForwardLighting() {}

void ForwardLighting::initRender()
{
    startTimer(0);
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

    // Point Lights
    PointLightData glPointLight;
    for (auto pl : pointLights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;
        //        glPointLight.position      = make_vec4(pl->getPosition(), 0.0f);
        glPointLight.colorDiffuse  = make_vec4(pl->getColorDiffuse(), pl->getIntensity());
        glPointLight.colorSpecular = make_vec4(pl->getColorSpecular(), 1.0f);  // specular Intensity?
        glPointLight.attenuation   = make_vec4(pl->getAttenuation(), pl->getRadius());
        ld.pointLights.push_back(glPointLight);
        li.pointLightCount++;
    }

    // Spot Lights
    SpotLightData glSpotLight;
    for (auto sl : spotLights)
    {
        if (li.spotLightCount >= maximumNumberOfSpotLights) break;  // just ignore too many lights...
        if (!sl->shouldRender()) continue;
        float cosa                = cos(radians(sl->getAngle() * 0.95f));  // make border smoother
        glSpotLight.position      = make_vec4(sl->getPosition(), cosa);
        glSpotLight.colorDiffuse  = make_vec4(sl->getColorDiffuse(), sl->getIntensity());
        glSpotLight.colorSpecular = make_vec4(sl->getColorSpecular(), 1.0f);  // specular Intensity?
        glSpotLight.attenuation   = make_vec4(sl->getAttenuation(), sl->getRadius());
        glSpotLight.direction     = make_vec4(0);
        glSpotLight.direction += sl->ModelMatrix().col(1);
        ld.spotLights.push_back(glSpotLight);
        li.spotLightCount++;
    }



    // Directional Lights
    DirectionalLightData glDirectionalLight;
    for (auto dl : directionalLights)
    {
        if (li.directionalLightCount >= maximumNumberOfDirectionalLights) break;  // just ignore too many lights...
        if (!dl->shouldRender()) continue;
        // glDirectionalLight.position      = make_vec4(dl->getPosition(), 0.0f);
        glDirectionalLight.colorDiffuse  = make_vec4(dl->getColorDiffuse(), dl->getIntensity());
        glDirectionalLight.colorSpecular = make_vec4(dl->getColorSpecular(), 1.0f);  // specular Intensity?
        glDirectionalLight.direction     = make_vec4(dl->getDirection(), 0.0f);
        ld.directionalLights.push_back(glDirectionalLight);
        li.directionalLightCount++;
    }

    lightDataBufferPoint.updateBuffer(ld.pointLights.data(), sizeof(PointLightData) * li.pointLightCount, 0);
    lightDataBufferSpot.updateBuffer(ld.spotLights.data(), sizeof(SpotLightData) * li.spotLightCount, 0);
    lightDataBufferDirectional.updateBuffer(ld.directionalLights.data(),
                                            sizeof(DirectionalLightData) * li.directionalLightCount, 0);

    lightInfoBuffer.updateBuffer(&li, sizeof(LightInfo), 0);
    visibleLights = li.pointLightCount + li.spotLightCount + li.directionalLightCount;
    stopTimer(0);
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
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLightData));
    maximumNumberOfPointLights = std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLightData));
    maximumNumberOfSpotLights  = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLightData));


    if (maximumNumberOfDirectionalLights != maxDirectionalLights)
    {
        lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLightData) * maxDirectionalLights,
                                                  GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfPointLights != maxPointLights)
    {
        lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLightData) * maxPointLights, GL_DYNAMIC_DRAW);
    }
    if (maximumNumberOfSpotLights != maxSpotLights)
    {
        lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLightData) * maxSpotLights, GL_DYNAMIC_DRAW);
    }

    maximumNumberOfDirectionalLights = maxDirectionalLights;
    maximumNumberOfPointLights       = maxPointLights;
    maximumNumberOfSpotLights        = maxSpotLights;
}

void ForwardLighting::renderImGui(bool* p_open)
{
    RendererLighting::renderImGui();
}

}  // namespace Saiga
