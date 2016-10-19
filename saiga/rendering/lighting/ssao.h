#pragma once

#include "saiga/camera/camera.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/rendering/postProcessor.h"

class SAIGA_GLOBAL SSAOShader : public DeferredShader{
public:
    GLint location_invProj;

    GLint location_randomImage;

    GLint location_kernelSize;
    GLint location_kernelOffsets;

    GLint location_radius;
    GLint location_power;

    float radius = 1.0f;
    float power = 1.0f;


    std::vector<vec3> kernelOffsets;


    virtual void checkUniforms();
    void uploadInvProj(mat4 &mat);
    void uploadData();
    void uploadRandomImage(Texture* img);
};

class SAIGA_GLOBAL SSAO{
public:
    bool ssao = false;
    MVPTextureShader* blurShader;
    SSAOShader* ssaoShader = nullptr;

    std::shared_ptr<Texture>  randomTexture;
    Texture* ssaotex = nullptr;

//    Texture* randomTexture;
    Texture* bluredTexture;

    Framebuffer ssao_framebuffer, ssao_framebuffer2;

    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;
    vec2 screenSize;
    glm::ivec2 ssaoSize;
    std::vector<vec3> kernelOffsets;

    SSAO(int w, int h);
    void resize(int w, int h);
    void clearSSAO();
    void render(Camera* cam, GBuffer *gbuffer);

    void setEnabled(bool enable);
    void toggle();

    void setKernelSize(int kernelSize);
    void setRadius(float radius){ssaoShader->radius = radius;}
    void setPower(float power){ssaoShader->power = power;}
};


