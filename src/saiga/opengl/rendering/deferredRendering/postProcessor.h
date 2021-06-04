/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/util/quality.h"
#include "saiga/opengl/UnifiedMeshBuffer.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/vertex.h"
namespace Saiga
{
class SAIGA_OPENGL_API PostProcessingShader : public Shader
{
   public:
    GLint location_texture, location_screenSize;
    GLint location_gbufferDepth, location_gbufferNormals, location_gbufferColor, location_gbufferMaterial;

    virtual void checkUniforms();
    virtual void uploadTexture(std::shared_ptr<TextureBase> texture);
    virtual void uploadGbufferTextures(GBuffer* gbuffer);
    virtual void uploadScreenSize(vec4 size);

    virtual void uploadAdditionalUniforms() {}
};

class SAIGA_OPENGL_API BrightnessShader : public PostProcessingShader
{
   public:
    GLint location_brightness;
    float brightness = 0.5f;

    void setBrightness(float b) { brightness = b; }

    virtual void checkUniforms();
    virtual void uploadAdditionalUniforms();
};


class SAIGA_OPENGL_API LightAccumulationShader : public DeferredShader
{
   public:
    GLint location_lightAccumulationtexture;

    virtual void checkUniforms();
    virtual void uploadLightAccumulationtexture(std::shared_ptr<TextureBase> texture);
};


struct SAIGA_OPENGL_API PostProcessorParameters
{
    Quality quality = Quality::LOW;
};

class SAIGA_OPENGL_API PostProcessor
{
   public:
    PostProcessor();
    void init(int width, int height, GBuffer* gbuffer, PostProcessorParameters params,
              std::shared_ptr<Texture> LightAccumulationTexture);

    void nextFrame();
    void bindCurrentBuffer();
    void switchBuffer();

    void render(bool use_gbuffer_as_first);

    void setPostProcessingEffects(const std::vector<std::shared_ptr<PostProcessingShader> >& postProcessingEffects);

    void resize(int width, int height);
    void blitLast(Framebuffer* target, ViewPort vp);
    void renderLast(Framebuffer* target, ViewPort vp);

    std::shared_ptr<Texture> getCurrentTexture();
    Framebuffer& getTargetBuffer();

   public:
    PostProcessorParameters params;
    int width, height;
    Framebuffer framebuffers[2];
    std::shared_ptr<Texture> textures[2];
    GBuffer* gbuffer;
    int currentBuffer = 0;
    int lastBuffer    = 1;
    UnifiedMeshBuffer quadMesh;
    std::vector<std::shared_ptr<PostProcessingShader> > postProcessingEffects;
    std::shared_ptr<PostProcessingShader> passThroughShader;

    std::shared_ptr<Shader> computeTest;

    // the first post processing shader reads from the lightaccumulation texture.
    std::shared_ptr<Texture> LightAccumulationTexture = nullptr;
    bool first                                        = false;

    void createFramebuffers();
    void applyShader(std::shared_ptr<PostProcessingShader> postProcessingShader);
};

}  // namespace Saiga
