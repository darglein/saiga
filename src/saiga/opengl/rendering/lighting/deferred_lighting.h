/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/vertex.h"

#include <set>

namespace Saiga
{
class SAIGA_OPENGL_API DeferredLighting : public RendererLighting
{
   public:
    int currentStencilId = 0;

    std::shared_ptr<Texture> ssaoTexture;

    std::shared_ptr<Texture> lightAccumulationTexture;
    Framebuffer lightAccumulationBuffer;

    std::shared_ptr<Texture> volumetricLightTexture, volumetricLightTexture2;
    Framebuffer  volumetricBuffer;

    DeferredLighting(GBuffer& gbuffer, GLTimerSystem* timer);
    DeferredLighting& operator=(DeferredLighting& l) = delete;
    ~DeferredLighting();

    void init(int width, int height, bool _useTimers) override;
    void resize(int width, int height) override;

    void loadShaders() override;

    //    std::shared_ptr<DirectionalLight> createDirectionalLight();
    //    std::shared_ptr<PointLight> createPointLight();
    //    std::shared_ptr<SpotLight> createSpotLight();

    void initRender() override;
    void render(Camera* cam, const ViewPort& viewPort) override;
    void postprocessVolumetric();

    void setStencilShader(std::shared_ptr<MVPShader> stencilShader);

    // add the volumetric light texture that was previously rendered to the scene
    void applyVolumetricLightBuffer();

    void renderImGui() override;


   public:
    std::shared_ptr<MVPTextureShader> textureShader;
    std::shared_ptr<MVPTextureShader> volumetricBlurShader;
    std::shared_ptr<Shader> volumetricBlurShader2;

    std::shared_ptr<PointLightShader> pointLightVolumetricShader;

    std::shared_ptr<SpotLightShader> spotLightVolumetricShader;


    ShaderPart::ShaderCodeInjections volumetricInjection;

    std::shared_ptr<MVPShader> stencilShader;
    GBuffer& gbuffer;

    bool stencilCulling = true;

    void blitGbufferDepthToAccumulationBuffer();
    void setupStencilPass();
    void setupLightPass(bool isVolumetric);

    template <typename T, typename shader_t>
    void renderLightVolume(lightMesh_t& mesh, T obj, Camera* cam, const ViewPort& vp, shader_t shader,
                           shader_t shaderShadow, shader_t shaderVolumetric);


    void renderDirectionalLights(Camera* cam, const ViewPort& vp, bool shadow);
};


template <typename T, typename shader_t>
inline void DeferredLighting::renderLightVolume(lightMesh_t& mesh, T obj, Camera* cam, const ViewPort& vp,
                                                shader_t shaderNormal, shader_t shaderShadow, shader_t shaderVolumetric)
{
    if (!obj->shouldRender()) return;

    if (stencilCulling)
    {
        setupStencilPass();
        if (stencilShader->bind())
        {
            stencilShader->uploadModel(obj->ModelMatrix());
            mesh.bindAndDraw();
            stencilShader->unbind();
        }
    }

    setupLightPass(obj->volumetric);
    shader_t shader = (obj->castShadows ? (obj->volumetric ? shaderVolumetric : shaderShadow) : shaderNormal);
    if (shader->bind())
    {
        shader->DeferredShader::uploadFramebuffer(&gbuffer);
        shader->uploadScreenSize(vp.getVec4());

        //    obj->bindUniforms(shader, cam);
        shader->SetUniforms(obj, cam);
        mesh.bindAndDraw();
        shader->unbind();
    }
}

}  // namespace Saiga
