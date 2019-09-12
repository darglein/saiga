/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/quality.h"
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
    GLint location_gbufferDepth, location_gbufferNormals, location_gbufferColor,
        location_gbufferData;  // depth and normal texture of gbuffer



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
    bool srgb       = false;  // colors stored in srgb.
    Quality quality = Quality::LOW;
};

class SAIGA_OPENGL_API PostProcessor
{
   private:
    PostProcessorParameters params;
    int width, height;
    Framebuffer framebuffers[2];
    std::shared_ptr<Texture> textures[2];
    GBuffer* gbuffer;
    int currentBuffer = 0;
    int lastBuffer    = 1;
    IndexedVertexBuffer<VertexNT, GLushort> quadMesh;
    std::vector<std::shared_ptr<PostProcessingShader> > postProcessingEffects;
    std::shared_ptr<PostProcessingShader> passThroughShader;

    bool useTimers = false;
    std::vector<FilteredMultiFrameOpenGLTimer> shaderTimer;

    std::shared_ptr<Shader> computeTest;

    // the first post processing shader reads from the lightaccumulation texture.
    std::shared_ptr<Texture> LightAccumulationTexture = nullptr;
    bool first                                        = false;

    void createFramebuffers();
    void applyShader(std::shared_ptr<PostProcessingShader> postProcessingShader);

   public:
    void createTimers();

    void init(int width, int height, GBuffer* gbuffer, PostProcessorParameters params,
              std::shared_ptr<Texture> LightAccumulationTexture, bool _useTimers);

    void nextFrame();
    void bindCurrentBuffer();
    void switchBuffer();

    void render();

    void setPostProcessingEffects(const std::vector<std::shared_ptr<PostProcessingShader> >& postProcessingEffects);

    void printTimings();
    void resize(int width, int height);
    void blitLast(int windowWidth, int windowHeight);
    void renderLast(int windowWidth, int windowHeight);

    framebuffer_texture_t getCurrentTexture();
    Framebuffer& getTargetBuffer();
};

}  // namespace Saiga
