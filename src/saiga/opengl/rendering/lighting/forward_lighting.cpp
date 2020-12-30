/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

#include "saiga/opengl/rendering/lighting/box_light.h"
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"

namespace Saiga
{
using namespace uber;

ForwardLighting::ForwardLighting() : RendererLighting()
{
    lightDataBufferDirectional.createGLBuffer(nullptr, sizeof(DirectionalLightData) * maximumNumberOfDirectionalLights,
                                              GL_DYNAMIC_DRAW);
    lightDataBufferPoint.createGLBuffer(nullptr, sizeof(PointLightData) * maximumNumberOfPointLights, GL_DYNAMIC_DRAW);
    lightDataBufferSpot.createGLBuffer(nullptr, sizeof(SpotLightData) * maximumNumberOfSpotLights, GL_DYNAMIC_DRAW);
    lightDataBufferBox.createGLBuffer(nullptr, sizeof(BoxLightData) * maximumNumberOfBoxLights, GL_DYNAMIC_DRAW);
    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);
}

ForwardLighting::~ForwardLighting() {}

void ForwardLighting::initRender()
{
    startTimer(0);
    RendererLighting::initRender();
    lightDataBufferPoint.bind(POINT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferSpot.bind(SPOT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferBox.bind(BOX_LIGHT_DATA_BINDING_POINT);
    lightDataBufferDirectional.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
    LightInfo li;
    LightData ld;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.boxLightCount         = 0;
    li.directionalLightCount = 0;

    // Point Lights
    PointLightData glPointLight;
    for (auto pl : pointLights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;
        glPointLight.position      = make_vec4(pl->getPosition(), 0.0f);
        glPointLight.colorDiffuse  = make_vec4(pl->getColorDiffuse(), pl->getIntensity());
        glPointLight.colorSpecular = make_vec4(pl->getColorSpecular(), 1.0f);  // specular Intensity?
        glPointLight.attenuation   = make_vec4(pl->getAttenuation(), pl->getRadius());
        ld.pointLights.push_back(glPointLight);
        li.pointLightCount++;
    }
    lightDataBufferPoint.updateBuffer(ld.pointLights.data(), sizeof(PointLightData) * li.pointLightCount, 0);

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
        glSpotLight.direction += sl->getModelMatrix().col(1);
        ld.spotLights.push_back(glSpotLight);
        li.spotLightCount++;
    }
    lightDataBufferSpot.updateBuffer(ld.spotLights.data(), sizeof(SpotLightData) * li.spotLightCount, 0);

    // Box Lights
    BoxLightData glBoxLight;
    const mat4 biasMatrix = make_mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0);
    for (auto bl : boxLights)
    {
        if (li.boxLightCount >= maximumNumberOfBoxLights) break;  // just ignore too many lights...
        if (!bl->shouldRender()) continue;
        glBoxLight.colorDiffuse  = make_vec4(bl->getColorDiffuse(), bl->getIntensity());
        glBoxLight.colorSpecular = make_vec4(bl->getColorSpecular(), 1.0f);  // specular Intensity?
        glBoxLight.direction     = make_vec4(0);
        glBoxLight.direction += bl->getModelMatrix().col(2);
        bl->calculateCamera();
        glBoxLight.lightMatrix = biasMatrix * bl->shadowCamera.proj * bl->shadowCamera.view;
        ld.boxLights.push_back(glBoxLight);
        li.boxLightCount++;
    }
    lightDataBufferBox.updateBuffer(ld.boxLights.data(), sizeof(BoxLightData) * li.boxLightCount, 0);

    // Directional Lights
    DirectionalLightData glDirectionalLight;
    for (auto dl : directionalLights)
    {
        if (li.directionalLightCount >= maximumNumberOfDirectionalLights) break;  // just ignore too many lights...
        if (!dl->shouldRender()) continue;
        glDirectionalLight.position      = make_vec4(dl->getPosition(), 0.0f);
        glDirectionalLight.colorDiffuse  = make_vec4(dl->getColorDiffuse(), dl->getIntensity());
        glDirectionalLight.colorSpecular = make_vec4(dl->getColorSpecular(), 1.0f);  // specular Intensity?
        glDirectionalLight.direction     = make_vec4(dl->getDirection(), 0.0f);
        ld.directionalLights.push_back(glDirectionalLight);
        li.directionalLightCount++;
    }
    lightDataBufferDirectional.updateBuffer(ld.directionalLights.data(),
                                            sizeof(DirectionalLightData) * li.directionalLightCount, 0);


    lightInfoBuffer.updateBuffer(&li, sizeof(LightInfo), 0);
    visibleLights = li.pointLightCount + li.spotLightCount + li.boxLightCount + li.directionalLightCount;
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

void ForwardLighting::setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights, int maxBoxLights)
{
    maxDirectionalLights = std::max(0, maxDirectionalLights);
    maxPointLights       = std::max(0, maxPointLights);
    maxSpotLights        = std::max(0, maxSpotLights);
    maxBoxLights         = std::max(0, maxBoxLights);

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
    if (maximumNumberOfBoxLights != maxBoxLights)
    {
        lightDataBufferBox.createGLBuffer(nullptr, sizeof(BoxLightData) * maxBoxLights, GL_DYNAMIC_DRAW);
    }

    maximumNumberOfDirectionalLights = maxDirectionalLights;
    maximumNumberOfPointLights       = maxPointLights;
    maximumNumberOfSpotLights        = maxSpotLights;
    maximumNumberOfBoxLights         = maxBoxLights;
}

void ForwardLighting::renderImGui(bool* p_open)
{
    RendererLighting::renderImGui();
}

}  // namespace Saiga