/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forward_lighting.h"

#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/rendering/lighting/cpu_plane_clusterer.h"
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/gpu_assignment_clusterer.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/six_plane_clusterer.h"
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

void ForwardLighting::init(int _width, int _height, bool _useTimers)
{
    RendererLighting::init(_width, _height, _useTimers);
    if (clustererType) lightClusterer->init(_width, _height);
}

void ForwardLighting::resize(int _width, int _height)
{
    RendererLighting::resize(_width, _height);
    if (clustererType) lightClusterer->resize(_width, _height);
}

void ForwardLighting::initRender()
{
    auto tim = timer->Measure("Light Init");
    RendererLighting::initRender();
    LightInfo li;
    LightData ld;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.directionalLightCount = 0;

    if (clustererType) lightClusterer->clearLightData();
    li.clusterEnabled = clustererType > 0;

    // Point Lights
    for (auto pl : pointLights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;
        ld.pointLights.push_back(pl->GetShaderData());

        if (clustererType) lightClusterer->addPointLight(pl->position, pl->radius);

        li.pointLightCount++;
    }

    // Spot Lights
    SpotLight::ShaderData glSpotLight;
    for (auto sl : spotLights)
    {
        if (li.spotLightCount >= maximumNumberOfSpotLights) break;  // just ignore too many lights...
        if (!sl->shouldRender()) continue;
        glSpotLight = sl->GetShaderData();
        ld.spotLights.push_back(glSpotLight);

        if (clustererType)
        {
            float rad = radians(sl->getAngle());
            float l   = sl->radius;
            float radius;
            if (rad > pi<float>() * 0.25f)
                radius = l * tan(rad);
            else
                radius = l * 0.5f / (cos(rad) * cos(rad));
            vec3 world_center =
                make_vec3(glSpotLight.position) + make_vec3(glSpotLight.direction).normalized() * radius;
            lightClusterer->addSpotLight(world_center, radius);
        }

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

    lightDataBufferPoint.bind(POINT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferSpot.bind(SPOT_LIGHT_DATA_BINDING_POINT);
    lightDataBufferDirectional.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
}

void ForwardLighting::cluster(Camera* cam, const ViewPort& viewPort)
{
    if (clustererType)
    {
        lightClusterer->clusterLights(cam, viewPort);
        // At this point we can use clustering information in the lighting uber shader with the right binding points.
    }
}

void ForwardLighting::render(Camera* cam, const ViewPort& viewPort)
{
    auto tim = timer->Measure("Light Render");
    // Does nothing
    RendererLighting::render(cam, viewPort);

    if (drawDebug)
    {
        //        glDepthMask(GL_TRUE);
        renderDebug(cam);
        //        glDepthMask(GL_FALSE);
    }
    if (clustererType) lightClusterer->renderDebug(cam);
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

    if (!showLightingImgui) return;
    ImGui::Begin("UberDefferedLighting", &showLightingImgui);


    const char* const clustererTypes[4] = {"None", "CPU SixPlanes", "CPU PlaneArrays", "GPU AABB Light Assignment"};

    bool changed = ImGui::Combo("Mode", &clustererType, clustererTypes, 4);

    if (changed)
    {
        setClusterType(clustererType);
    }
    ImGui::End();

    if (clustererType) lightClusterer->renderImGui();
}

void ForwardLighting::setClusterType(int tp)
{
    clustererType = tp;
    if (clustererType > 0)
    {
        switch (clustererType)
        {
            case 1:
                lightClusterer =
                    std::static_pointer_cast<Clusterer>(std::make_shared<SixPlaneClusterer>(timer));
                break;
            case 2:
                lightClusterer =
                    std::static_pointer_cast<Clusterer>(std::make_shared<CPUPlaneClusterer>(timer));
                break;
            case 3:
                lightClusterer =
                    std::static_pointer_cast<Clusterer>(std::make_shared<GPUAssignmentClusterer>(timer));
                break;
            default:
                lightClusterer = nullptr;
                return;
        }

        lightClusterer->init(width, height);
    }
}

}  // namespace Saiga
