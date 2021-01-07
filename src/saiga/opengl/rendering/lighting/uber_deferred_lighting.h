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
#include "saiga/opengl/rendering/lighting/light_clusterer.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"
#ifdef SHADER_STORAGE_BUFFER
#    include "saiga/opengl/shaderStorageBuffer.h"
#endif
#include "saiga/opengl/vertex.h"

#include <set>

namespace Saiga
{
#ifdef SHADER_STORAGE_BUFFER
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
#else
class SAIGA_OPENGL_API UberDeferredLightingShader : public DeferredShader
{
   public:
    GLint location_lightDataBlockPoint;
    GLint location_lightDataBlockSpot;
    GLint location_lightDataBlockBox;
    GLint location_lightDataBlockDirectional;
    GLint location_lightInfoBlock;

    GLint location_invProj;

    virtual void checkUniforms() override
    {
        DeferredShader::checkUniforms();
        location_lightDataBlockPoint = getUniformBlockLocation("lightDataBlockPoint");
        if (location_lightDataBlockPoint != GL_INVALID_INDEX)
            setUniformBlockBinding(location_lightDataBlockPoint, POINT_LIGHT_DATA_BINDING_POINT);
        location_lightDataBlockSpot = getUniformBlockLocation("lightDataBlockSpot");
        if (location_lightDataBlockSpot != GL_INVALID_INDEX)
            setUniformBlockBinding(location_lightDataBlockSpot, SPOT_LIGHT_DATA_BINDING_POINT);
        location_lightDataBlockBox = getUniformBlockLocation("lightDataBlockBox");
        if (location_lightDataBlockBox != GL_INVALID_INDEX)
            setUniformBlockBinding(location_lightDataBlockBox, BOX_LIGHT_DATA_BINDING_POINT);
        location_lightDataBlockDirectional = getUniformBlockLocation("lightDataBlockDirectional");
        if (location_lightDataBlockDirectional != GL_INVALID_INDEX)
            setUniformBlockBinding(location_lightDataBlockDirectional, DIRECTIONAL_LIGHT_DATA_BINDING_POINT);


        location_lightInfoBlock = getUniformBlockLocation("lightInfoBlock");
        setUniformBlockBinding(location_lightInfoBlock, LIGHT_INFO_BINDING_POINT);


        location_invProj = getUniformLocation("invProj");
    }

    inline void uploadInvProj(const mat4& mat) { Shader::upload(location_invProj, mat); }
};
#endif

class SAIGA_OPENGL_API UberDeferredLighting : public RendererLighting
{
   public:
#ifdef SHADER_STORAGE_BUFFER
    ShaderStorageBuffer lightDataBufferPoint;
    ShaderStorageBuffer lightDataBufferSpot;
    ShaderStorageBuffer lightDataBufferBox;
    ShaderStorageBuffer lightDataBufferDirectional;
#else
    UniformBuffer lightDataBufferPoint;
    UniformBuffer lightDataBufferSpot;
    UniformBuffer lightDataBufferBox;
    UniformBuffer lightDataBufferDirectional;
#endif
    UniformBuffer lightInfoBuffer;

    UberDeferredLighting(GBuffer& gbuffer);
    UberDeferredLighting& operator=(UberDeferredLighting& l) = delete;
    ~UberDeferredLighting();

    void loadShaders() override;
    void init(int _width, int _height, bool _useTimers) override;
    void resize(int width, int height) override;

    void initRender() override;
    void render(Camera* cam, const ViewPort& viewPort) override;

    void renderImGui(bool* p_open = NULL) override;

    void setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights, int maxBoxLights) override;

   public:
    std::shared_ptr<UberDeferredLightingShader> lightingShader;
    GBuffer& gbuffer;
    IndexedVertexBuffer<VertexNT, GLushort> quadMesh;
    std::shared_ptr<Clusterer> lightClusterer;
};

}  // namespace Saiga