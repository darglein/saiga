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
#include "saiga/opengl/rendering/lighting/gpu_aabb_clusterer.h"
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

    lightInfoBuffer.createGLBuffer(nullptr, sizeof(LightInfo), GL_DYNAMIC_DRAW);

    lightClusterer                 = std::make_shared<CPUPlaneClusterer>(timer);
}

void UberDeferredLighting::init(int _width, int _height, bool _useTimers)
{
    RendererLighting::init(_width, _height, _useTimers);
    if (clustererType) lightClusterer->resize(_width, _height);

    lightAccumulationBuffer.create();

    lightAccumulationBuffer.attachTextureDepthStencil(gbuffer.getTextureDepth());

    lightAccumulationTexture = std::make_shared<Texture>();
    lightAccumulationTexture->create(_width, _height, GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);
    lightAccumulationBuffer.attachTexture(lightAccumulationTexture);

    lightAccumulationBuffer.drawTo({0});
    lightAccumulationBuffer.check();
    lightAccumulationBuffer.unbind();
}

void UberDeferredLighting::resize(int _width, int _height)
{
    RendererLighting::resize(_width, _height);
    if (clustererType) lightClusterer->resize(_width, _height);
    lightAccumulationBuffer.resize(_width, _height);
}

UberDeferredLighting::~UberDeferredLighting() {}

void UberDeferredLighting::loadShaders()
{
    RendererLighting::loadShaders();

    const RendererLightingShaderNames& names = RendererLightingShaderNames();

    lightingShader = shaderLoader.load<UberDeferredLightingShader>(names.lightingUberShader);
}

void UberDeferredLighting::initRender()
{
    auto tim = timer->Measure("Lightinit");
    RendererLighting::initRender();
    LightInfo li;
    li.pointLightCount       = 0;
    li.spotLightCount        = 0;
    li.directionalLightCount = 0;


    li.clusterEnabled = clustererType > 0;

    // Point Lights
    li.pointLightCount = active_point_lights.size();

    // Spot Lights
    li.spotLightCount = active_spot_lights.size();

    // Directional Lights
    li.directionalLightCount = active_directional_lights.size();

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
        lightClusterer->clusterLights(cam, viewPort, active_point_lights, active_spot_lights);
        // At this point we can use clustering information in the lighting uber shader with the right binding points.
    }


    lightAccumulationBuffer.bind();
    {
        auto tim = timer->Measure("Shade");
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Lighting Uber Shader
        if(lightingShader->bind())
        {
            lightingShader->uploadFramebuffer(&gbuffer);
            lightingShader->uploadScreenSize(viewPort.getVec4());
            lightingShader->uploadInvProj(inverse(cam->proj));
            quadMesh.BindAndDraw();
            lightingShader->unbind();
        }
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

    lightAccumulationBuffer.unbind();
    assert_no_glerror();
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

    if (clustererType) lightClusterer->imgui();
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
                lightClusterer = std::static_pointer_cast<Clusterer>(std::make_shared<GPUAABBClusterer>(timer));
                break;
            default:
                lightClusterer = nullptr;
                return;
        }

        lightClusterer->resize(width, height);
    }
}

}  // namespace Saiga
