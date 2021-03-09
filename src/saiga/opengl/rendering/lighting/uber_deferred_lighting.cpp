/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "uber_deferred_lighting.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
using namespace uber;

UberDeferredLighting::UberDeferredLighting(GBuffer& framebuffer) : gbuffer(framebuffer)
{
    createLightMeshes();
    shadowCameraBuffer.createGLBuffer(nullptr, sizeof(CameraDataGLSL), GL_DYNAMIC_DRAW);

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


    quadMesh.fromMesh(FullScreenQuad());

    ClustererParameters params;
    lightClusterer = std::make_shared<Clusterer>(params);
}

void UberDeferredLighting::init(int _width, int _height, bool _useTimers)
{
    RendererLighting::init(_width, _height, _useTimers);
    if (lightClusterer) lightClusterer->init(_width, _height, _useTimers);
}

void UberDeferredLighting::resize(int _width, int _height)
{
    RendererLighting::resize(_width, _height);
    if (lightClusterer) lightClusterer->resize(_width, _height);
}

UberDeferredLighting::~UberDeferredLighting() {}

void UberDeferredLighting::loadShaders()
{
    RendererLighting::loadShaders();

    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    ShaderPart::ShaderCodeInjections sci;

    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfDirectionalLights), 1);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfPointLights), 2);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfSpotLights), 3);
    lightingShader = shaderLoader.load<UberDeferredLightingShader>(names.lightingUberShader, sci);
}

void UberDeferredLighting::initRender()
{
    // TODO Paul: We should refactor this for all single light pass renderers.
    startTimer(0);
    RendererLighting::initRender();
    LightInfo li;
    LightData ld;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.directionalLightCount = 0;

    if (lightClustererEnabled) lightClusterer->clearLightData();

    // Point Lights
    for (auto pl : pointLights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;
        ld.pointLights.push_back(pl->GetShaderData());

        if (lightClustererEnabled) lightClusterer->addPointLight(pl->position, pl->radius);

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

        if (lightClustererEnabled)
        {
            float rad = radians(sl->getAngle());
            float l   = tan(rad) * sl->radius;
            float radius;
            if (rad > pi<float>() * 0.25f)
                radius = l * tan(rad);
            else
                radius = l * 0.5f / pow(cos(sl->getAngle()), 2.0f);
            vec3 world_center =
                sl->getPosition() +
                vec3(glSpotLight.direction.x(), glSpotLight.direction.y(), glSpotLight.direction.z()) * radius;
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
    lightDataBufferBox.bind(BOX_LIGHT_DATA_BINDING_POINT);
    lightDataBufferDirectional.bind(DIRECTIONAL_LIGHT_DATA_BINDING_POINT);
    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
    stopTimer(0);
}

void UberDeferredLighting::render(Camera* cam, const ViewPort& viewPort)
{
    // Does nothing
    RendererLighting::render(cam, viewPort);
    if (lightClustererEnabled)
    {
        lightClusterer->clusterLights(cam, viewPort);
        // At this point we can use clustering information in the lighting uber shader with the right binding points.
    }


    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Lighting Uber Shader
    lightingShader->bind();
    lightingShader->uploadFramebuffer(&gbuffer);
    lightingShader->uploadScreenSize(viewPort.getVec4());
    lightingShader->uploadInvProj(inverse(cam->proj));
    quadMesh.bindAndDraw();
    lightingShader->unbind();
    assert_no_glerror();

    // reset state
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    if (drawDebug)
    {
        // glDepthMask(GL_TRUE);
        renderDebug(cam);
        // glDepthMask(GL_FALSE);
    }
    if(lightClustererEnabled)
        lightClusterer->renderDebug(cam);
    assert_no_glerror();
}

void UberDeferredLighting::setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights)
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

    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    ShaderPart::ShaderCodeInjections sci;
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_DL_COUNT" + std::to_string(maximumNumberOfDirectionalLights), 1);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_PL_COUNT" + std::to_string(maximumNumberOfPointLights), 2);
    sci.emplace_back(GL_FRAGMENT_SHADER, "#define MAX_SL_COUNT" + std::to_string(maximumNumberOfSpotLights), 3);
    lightingShader = shaderLoader.load<UberDeferredLightingShader>(names.lightingUberShader, sci);
}

void UberDeferredLighting::renderImGui()
{
    RendererLighting::renderImGui(p_open);
    ImGui::Begin("UberDefferedLighting", p_open);
    bool changed          = ImGui::Checkbox("lightClustererEnabled", &lightClustererEnabled);
    if (changed)
    {
        if (lightClustererEnabled)
            lightClusterer->enable();
        else
            lightClusterer->disable();
    }
    ImGui::End();

    if (lightClustererEnabled) lightClusterer->renderImGui();
}

}  // namespace Saiga
