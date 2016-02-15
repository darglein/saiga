#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/util/quality.h"

class SAIGA_GLOBAL PostProcessingShader : public Shader{
public:
    GLint location_texture, location_screenSize;
	GLint location_gbufferDepth, location_gbufferNormals, location_gbufferColor; //depth and normal texture of gbuffer



    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
    virtual void uploadGbufferTextures(GBuffer* gbuffer);
    virtual void uploadScreenSize(vec4 size);

    virtual void uploadAdditionalUniforms(){}
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
    IndexedVertexBuffer<VertexNT,GLubyte> quadMesh;
    std::vector<PostProcessingShader*> postProcessingEffects;
    GPUTimer timer;

    Shader* computeTest;

    void createFramebuffers();
    void applyShader(PostProcessingShader* postProcessingShader);
    void applyShaderFinal(PostProcessingShader *postProcessingShader);
public:

    void init(int width, int height, GBuffer *gbuffer, PostProcessorParameters params);

    void nextFrame();
    void bindCurrentBuffer();
    void switchBuffer();

    void render();

    void setPostProcessingEffects(const std::vector<PostProcessingShader*> &postProcessingEffects ){this->postProcessingEffects = postProcessingEffects;}


    void resize(int width, int height);
};
