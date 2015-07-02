#pragma once

#include "libhello/rendering/renderer.h"
#include "libhello/rendering/lighting/deferred_lighting.h"
#include "libhello/opengl/framebuffer.h"
//#include "libhello/opengl/mesh.h"


class PostProcessingShader : public Shader{
public:
    GLuint location_texture, location_screenSize;
    PostProcessingShader(const std::string &multi_file) : Shader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
    virtual void uploadScreenSize(vec4 size);
    virtual void uploadAdditionalUniforms(){}
};



class LightAccumulationShader : public DeferredShader{
public:
    GLuint location_lightAccumulationtexture;
    LightAccumulationShader(const std::string &multi_file) : DeferredShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadLightAccumulationtexture(raw_Texture* texture);
};

class PostProcessor{
    int width,height;
    Framebuffer framebuffers[2];
    Texture* textures[2];
    int currentBuffer = 0;
    int lastBuffer = 1;
    IndexedVertexBuffer<VertexNT,GLuint> quadMesh;
    std::vector<PostProcessingShader*> postProcessingEffects;

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


};
