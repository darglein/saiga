#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/util/quality.h"

class SAIGA_GLOBAL PostProcessingShader : public Shader{
public:
    GLint location_texture, location_screenSize;
    GLint location_gbufferDepth, location_gbufferNormals, location_gbufferColor, location_gbufferData; //depth and normal texture of gbuffer



    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
    virtual void uploadGbufferTextures(GBuffer* gbuffer);
    virtual void uploadScreenSize(vec4 size);

    virtual void uploadAdditionalUniforms(){}
};

class SAIGA_GLOBAL BrightnessShader : public PostProcessingShader{
public:
    GLint location_brightness;
    float brightness = 0.5f;

    void setBrightness(float b){brightness=b;}

    virtual void checkUniforms();
    virtual void uploadAdditionalUniforms();
};


class SAIGA_GLOBAL LightAccumulationShader : public DeferredShader{
public:
    GLint location_lightAccumulationtexture;

    virtual void checkUniforms();
    virtual void uploadLightAccumulationtexture(raw_Texture* texture);
};


struct SAIGA_GLOBAL PostProcessorParameters{
    bool srgb = true; //colors stored in srgb. saves memory bandwith but adds conversion operations.
    Quality quality = Quality::LOW;
};

class SAIGA_GLOBAL PostProcessor{
private:
    PostProcessorParameters params;
    int width,height;
    Framebuffer framebuffers[2];
    Texture* textures[2];
    GBuffer *gbuffer;
    int currentBuffer = 0;
    int lastBuffer = 1;
    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;
    std::vector<PostProcessingShader*> postProcessingEffects;

    bool useTimers = false;
    std::vector<FilteredGPUTimer> shaderTimer;

    Shader* computeTest;

    //the first post processing shader reads from the lightaccumulation texture.
    Texture* LightAccumulationTexture = nullptr;
    bool first = false;

    void createFramebuffers();
    void applyShader(PostProcessingShader* postProcessingShader);
    void applyShaderFinal(PostProcessingShader *postProcessingShader);
public:

    void init(int width, int height, GBuffer *gbuffer, PostProcessorParameters params, Texture* LightAccumulationTexture, bool _useTimers);

    void nextFrame();
    void bindCurrentBuffer();
    void switchBuffer();

    void render();

    void setPostProcessingEffects(const std::vector<PostProcessingShader*> &postProcessingEffects );

    void printTimings();
    void resize(int width, int height);
    void blitLast(int windowWidth, int windowHeight);

    framebuffer_texture_t getCurrentTexture();
};
