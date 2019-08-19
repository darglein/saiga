/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/rendering/deferredRendering/postProcessor.h"

namespace Saiga
{
class SAIGA_OPENGL_API SSAOShader : public DeferredShader
{
   public:
    GLint location_invProj;

    GLint location_randomImage;

    GLint location_kernelSize;
    GLint location_kernelOffsets;

    GLint location_radius;
    GLint location_power;

    float radius   = 1.0f;
    float exponent = 1.0f;


    std::vector<vec3> kernelOffsets;


    virtual void checkUniforms();
    void uploadInvProj(mat4& mat);
    void uploadData();
    void uploadRandomImage(std::shared_ptr<Texture> img);
};

class SAIGA_OPENGL_API SSAO
{
   private:
    std::shared_ptr<MVPTextureShader> blurShader;
    std::shared_ptr<SSAOShader> ssaoShader = nullptr;

    std::shared_ptr<Texture> randomTexture;
    Framebuffer ssao_framebuffer, ssao_framebuffer2;
    std::shared_ptr<Texture> ssaotex;

    IndexedVertexBuffer<VertexNT, GLushort> quadMesh;
    vec2 screenSize;
    ivec2 ssaoSize;
    std::vector<vec3> kernelOffsets;
    int kernelSize = 32;

   public:
    std::shared_ptr<Texture> bluredTexture;

    SSAO(int w, int h);
    void init(int w, int h);
    void resize(int w, int h);
    void clearSSAO();
    void render(Camera* cam, const ViewPort &vp, GBuffer* gbuffer);


    void setKernelSize(int kernelSize);

    void renderImGui();
};

}  // namespace Saiga
