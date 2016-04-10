#pragma once

#include "saiga/camera/camera.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"

class PointLightShader;
class SpotLightShader;
class DirectionalLightShader;
class BoxLightShader;
class LightAccumulationShader;

class SpotLight;
class PointLight;
class DirectionalLight;
class BoxLight;

class Program;


class SAIGA_GLOBAL DeferredLighting{
    friend class LightingController;
private:
    int width,height;
    MVPColorShader* debugShader;

    //the vertex position is sufficient. no normals and texture coordinates needed.
    typedef IndexedVertexBuffer<Vertex,GLushort> lightMesh_t;

    PointLightShader* pointLightShader, *pointLightShadowShader;
    lightMesh_t pointLightMesh;
    std::vector<PointLight*> pointLights;

    SpotLightShader* spotLightShader, *spotLightShadowShader;
    lightMesh_t spotLightMesh;
    std::vector<SpotLight*> spotLights;

    DirectionalLightShader* directionalLightShader,*directionalLightShadowShader;
    lightMesh_t directionalLightMesh;
    std::vector<DirectionalLight*> directionalLights;

    BoxLightShader* boxLightShader,*boxLightShadowShader;
    lightMesh_t boxLightMesh;
    std::vector<BoxLight*> boxLights;

    MVPTextureShader* blitDepthShader;
    LightAccumulationShader* lightAccumulationShader;

    MVPShader* stencilShader;
    GBuffer &gbuffer;


    mat4 inview,view,proj;
    bool drawDebug = true;

//    raw_Texture* dummyTexture;
//    raw_Texture* dummyCubeTexture;


public:
    int totalLights;
    int visibleLights;
    int renderedDepthmaps;
    int currentStencilId = 0;

    Texture* ssaoTexture;

    Texture* lightAccumulationTexture;
    Framebuffer lightAccumulationBuffer;

    DeferredLighting(GBuffer &gbuffer);
	DeferredLighting& operator=(DeferredLighting& l) = delete;
    ~DeferredLighting();

    void init(int width, int height);
    void resize(int width, int height);

    void loadShaders();

    void setRenderDebug(bool b){drawDebug = b;}
    void createLightMeshes();

    DirectionalLight* createDirectionalLight();
    PointLight* createPointLight();
    SpotLight* createSpotLight();
    BoxLight* createBoxLight();

    void removeDirectionalLight(DirectionalLight* l);
    void removePointLight(PointLight* l);
    void removeSpotLight(SpotLight* l);
    void removeBoxLight(BoxLight* l);


    void render(Camera *cam);
    void renderLightAccumulation();
    void renderDepthMaps(Program *renderer );
    void renderDebug();


    void setShader(SpotLightShader* spotLightShader, SpotLightShader* spotLightShadowShader);
    void setShader(PointLightShader* pointLightShader,PointLightShader* pointLightShadowShader);
    void setShader(DirectionalLightShader* directionalLightShader,DirectionalLightShader* directionalLightShadowShader);
    void setShader(BoxLightShader* boxLightShader,BoxLightShader* boxLightShadowShader);

    void setDebugShader(MVPColorShader* shader);
    void setStencilShader(MVPShader* stencilShader);


    void setViewProj(const mat4 &iv,const mat4 &v,const mat4 &p);

    void cullLights(Camera *cam);
private:
    void renderPointLight(PointLight* p, Camera *cam);

    void createInputCommands();

    void blitGbufferDepthToAccumulationBuffer();
    void setupStencilPass();
    void setupLightPass();

    template<typename T>
    void renderStencilVolume(lightMesh_t &mesh, std::vector<T*> &objs);

    template<typename T,typename shader_t, bool shadow>
    void renderLightVolume(lightMesh_t &mesh, std::vector<T *> &objs, Camera *cam, shader_t *shader);

    template<typename T,typename shader_t>
    void renderLightVolume(lightMesh_t &mesh, T* obj, Camera *cam, shader_t *shader , shader_t *shaderShadow);


    void renderPointLights(Camera *cam, bool shadow);
    void renderPointLightsStencil();

    void renderSpotLights(Camera *cam, bool shadow);
    void renderSpotLightsStencil();

    void renderBoxLights(Camera *cam, bool shadow);
    void renderBoxLightsStencil();

    void renderDirectionalLights(Camera *cam, bool shadow);


};


template<typename T,typename shader_t>
inline void DeferredLighting::renderLightVolume(lightMesh_t &mesh, T* obj, Camera *cam, shader_t *shaderNormal , shader_t *shaderShadow){
    if(!obj->shouldRender())
        return;

    setupStencilPass();
    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);

    obj->bindUniformsStencil(*stencilShader);
    mesh.bindAndDraw();
    stencilShader->unbind();


    setupLightPass();
    shader_t* shader = (obj->hasShadows()) ? shaderShadow : shaderNormal;
    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&gbuffer);
    shader->uploadScreenSize(vec2(width,height));

    obj->bindUniforms(*shader,cam);
    mesh.bindAndDraw();
    shader->unbind();


}
