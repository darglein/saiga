/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/lighting/clusterer.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/vertex.h"

#include <set>

namespace Saiga
{
class SAIGA_OPENGL_API UberDeferredLightingShader : public DeferredShader
{
   public:
    GLint location_lightInfoBlock;

    GLint location_invProj;

    virtual void checkUniforms() override
    {
        DeferredShader::checkUniforms();

        location_lightInfoBlock = getUniformBlockLocation("lightInfoBlock");
        setUniformBlockBinding(location_lightInfoBlock, LIGHT_INFO_BINDING_POINT);


        location_invProj = getUniformLocation("invProj");
    }

    inline void uploadInvProj(const mat4& mat) { Shader::upload(location_invProj, mat); }
};

class SAIGA_OPENGL_API UberDeferredLighting : public RendererLighting
{
   public:

    UniformBuffer lightInfoBuffer;

    std::shared_ptr<Texture> lightAccumulationTexture;
    Framebuffer lightAccumulationBuffer;

    UberDeferredLighting(GBuffer& gbuffer, GLTimerSystem* timer);
    UberDeferredLighting& operator=(UberDeferredLighting& l) = delete;
    ~UberDeferredLighting();

    void loadShaders() override;
    void init(int _width, int _height, bool _useTimers) override;
    void resize(int width, int height) override;

    void initRender() override;
    void render(Camera* cam, const ViewPort& viewPort) override;

    void renderImGui() override;

    void setClusterType(int tp) override;

    std::shared_ptr<Clusterer> getClusterer() override { return lightClusterer; };

   public:
    std::shared_ptr<UberDeferredLightingShader> lightingShader;
    GBuffer& gbuffer;
    UnifiedMeshBuffer quadMesh;
    std::shared_ptr<Clusterer> lightClusterer;
    int clustererType = 0;
};

}  // namespace Saiga
