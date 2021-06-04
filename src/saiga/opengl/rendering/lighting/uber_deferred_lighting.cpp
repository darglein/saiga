/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "uber_deferred_lighting.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/lighting/cpu_plane_clusterer.h"
#include "saiga/opengl/rendering/lighting/gpu_assignment_clusterer.h"
#include "saiga/opengl/rendering/lighting/six_plane_clusterer.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/CubeTexture.h"

namespace Saiga
{
using namespace uber;

UberDeferredLighting::UberDeferredLighting(GBuffer& framebuffer, GLTimerSystem* timer)
    : RendererLighting(timer), gbuffer(framebuffer),  quadMesh(FullScreenQuad())
{
    createLightMeshes();

    int maxSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();

    maximumNumberOfDirectionalLights =
        std::clamp(maximumNumberOfDirectionalLights, 0, maxSize / (int)sizeof(DirectionalLight::ShaderData));
    maximumNumberOfPointLights =
        std::clamp(maximumNumberOfPointLights, 0, maxSize / (int)sizeof(PointLight::ShaderData));
    maximumNumberOfSpotLights = std::clamp(maximumNumberOfSpotLights, 0, maxSize / (int)sizeof(SpotLight::ShaderData));

    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);

    lightClusterer                 = std::make_shared<CPUPlaneClusterer>(timer);
}

void UberDeferredLighting::init(int _width, int _height, bool _useTimers)
{
    RendererLighting::init(_width, _height, _useTimers);
    if (clustererType) lightClusterer->init(_width, _height);
}

void UberDeferredLighting::resize(int _width, int _height)
{
    RendererLighting::resize(_width, _height);
    if (clustererType) lightClusterer->resize(_width, _height);
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
    auto tim = timer->Measure("Lightinit");
    RendererLighting::initRender();
    LightInfo li;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.directionalLightCount = 0;

    if (clustererType) lightClusterer->clearLightData();
    li.clusterEnabled = clustererType > 0;

    // Point Lights
    for (auto& pl : active_point_lights)
    {
        if (li.pointLightCount >= maximumNumberOfPointLights) break;  // just ignore too many lights...
        if (!pl->shouldRender()) continue;

        if (clustererType) lightClusterer->addPointLight(pl->position, pl->radius);

        li.pointLightCount++;
    }

    // Spot Lights
    for (auto& sl : active_spot_lights)
    {
        if (li.spotLightCount >= maximumNumberOfSpotLights) break;  // just ignore too many lights...
        if (!sl->shouldRender()) continue;

        if (clustererType)
        {
            float rad = radians(sl->getAngle());
            float l   = sl->radius;
            float radius;
            if (rad > pi<float>() * 0.25f)
                radius = l * tan(rad);
            else
                radius = l * 0.5f / (cos(rad) * cos(rad));
            vec3 world_center = sl->position + sl->direction.normalized() * radius;
            lightClusterer->addSpotLight(world_center, radius);
        }

        li.spotLightCount++;
    }

    // Directional Lights
    for (auto& dl : active_directional_lights)
    {
        if (li.directionalLightCount >= maximumNumberOfDirectionalLights) break;  // just ignore too many lights...
        if (!dl->shouldRender()) continue;
        li.directionalLightCount++;
    }

    lightInfoBuffer.updateBuffer(&li, sizeof(LightInfo), 0);
    visibleLights = li.pointLightCount + li.spotLightCount + li.directionalLightCount;

    lightInfoBuffer.bind(LIGHT_INFO_BINDING_POINT);
}

void UberDeferredLighting::render(Camera* cam, const ViewPort& viewPort)
{
    // Does nothing
    RendererLighting::render(cam, viewPort);
    if (clustererType)
    {
        lightClusterer->clusterLights(cam, viewPort);
        // At this point we can use clustering information in the lighting uber shader with the right binding points.
    }


    {
        auto tim = timer->Measure("Shade");
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Lighting Uber Shader
        lightingShader->bind();
        lightingShader->uploadFramebuffer(&gbuffer);
        lightingShader->uploadScreenSize(viewPort.getVec4());
        lightingShader->uploadInvProj(inverse(cam->proj));
        quadMesh.BindAndDraw();
        lightingShader->unbind();
        assert_no_glerror();

        // reset state
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    if (drawDebug)
    {
        // glDepthMask(GL_TRUE);
        renderDebug(cam);
        // glDepthMask(GL_FALSE);
    }
    if (clustererType) lightClusterer->renderDebug(cam);
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
    if (!showLightingImgui) return;

    if (!editor_gui.enabled)
    {
        int w = 340;
        int h = 240;
        ImGui::SetNextWindowPos(ImVec2(680, height - h), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_Once);
    }

    if (ImGui::Begin("Lighting", &showLightingImgui))
    {
        const char* const clustererTypes[4] = {"None", "CPU SixPlanes", "CPU PlaneArrays", "GPU AABB Light Assignment"};

        bool changed = ImGui::Combo("Mode", &clustererType, clustererTypes, 4);

        if (changed)
        {
            setClusterType(clustererType);
        }
        ImGui::Separator();
    }
    ImGui::End();

    RendererLighting::renderImGui();

    if (clustererType) lightClusterer->renderImGui();
}

void UberDeferredLighting::setClusterType(int tp)
{
    clustererType = tp;
    if (clustererType > 0)
    {
        switch (clustererType)
        {
            case 1:
                lightClusterer = std::static_pointer_cast<Clusterer>(std::make_shared<SixPlaneClusterer>(timer));
                break;
            case 2:
                lightClusterer = std::static_pointer_cast<Clusterer>(std::make_shared<CPUPlaneClusterer>(timer));
                break;
            case 3:
                lightClusterer = std::static_pointer_cast<Clusterer>(std::make_shared<GPUAssignmentClusterer>(timer));
                break;
            default:
                lightClusterer = nullptr;
                return;
        }

        lightClusterer->init(width, height);
    }
}

}  // namespace Saiga
