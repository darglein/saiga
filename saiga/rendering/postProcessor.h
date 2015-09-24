#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"


class SAIGA_GLOBAL PostProcessingShader : public Shader{
public:
    GLint location_texture, location_screenSize;

    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
    virtual void uploadScreenSize(vec4 size);
    virtual void uploadAdditionalUniforms(){}
};



class SAIGA_GLOBAL LightAccumulationShader : public DeferredShader{
public:
    GLint location_lightAccumulationtexture;

    virtual void checkUniforms();
    virtual void uploadLightAccumulationtexture(raw_Texture* texture);
};



class SAIGA_GLOBAL PostProcessor{
private:
    int width,height;
    Framebuffer framebuffers[2];
    Texture* textures[2];
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

    void init(int width, int height);

    void nextFrame(Framebuffer* gbuffer);
    void bindCurrentBuffer();
    void switchBuffer();

    void render();

    void setPostProcessingEffects(const std::vector<PostProcessingShader*> &postProcessingEffects ){this->postProcessingEffects = postProcessingEffects;}


    void resize(int width, int height);
};
