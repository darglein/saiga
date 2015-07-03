#pragma once

#include "libhello/rendering/renderer.h"
#include "libhello/opengl/framebuffer.h"
#include "libhello/opengl/mesh.h"

class SpotLight;
class PointLight;
class DirectionalLight;
class BoxLight;
class PointLightShader;
class SpotLightShader;
class DirectionalLightShader;
class BoxLightShader;
class LightAccumulationShader;

class Deferred_Renderer;

class SAIGA_GLOBAL DeferredLighting{
    friend class LightingController;
private:
    int width,height;
    MVPColorShader* debugShader;

    PointLightShader* pointLightShader, *pointLightShadowShader;
    IndexedVertexBuffer<VertexNT,GLuint> pointLightMesh;
    std::vector<PointLight*> pointLights;

    SpotLightShader* spotLightShader, *spotLightShadowShader;
    IndexedVertexBuffer<VertexNT,GLuint> spotLightMesh;
    std::vector<SpotLight*> spotLights;

    DirectionalLightShader* directionalLightShader,*directionalLightShadowShader;
    IndexedVertexBuffer<VertexNT,GLuint> directionalLightMesh;
    std::vector<DirectionalLight*> directionalLights;

    BoxLightShader* boxLightShader,*boxLightShadowShader;
    IndexedVertexBuffer<VertexNT,GLuint> boxLightMesh;
    std::vector<BoxLight*> boxLights;

    LightAccumulationShader* lightAccumulationShader;

    MVPShader* stencilShader;
    Framebuffer &gbuffer;


    mat4 inview,view,proj;
    bool drawDebug = true;

//    raw_Texture* dummyTexture;
//    raw_Texture* dummyCubeTexture;


public:
    int totalLights;
    int visibleLights;
    int renderedDepthmaps;

    Texture* ssaoTexture;

    Texture* lightAccumulationTexture;
    Framebuffer lightAccumulationBuffer;

    DeferredLighting(Framebuffer &gbuffer);
    ~DeferredLighting();

    void init(int width, int height);

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
    void renderDepthMaps(RendererInterface *renderer );
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
    void createInputCommands();

    void setupStencilPass();
    void setupLightPass();

    template<typename T>
    void renderStencilVolume(IndexedVertexBuffer<VertexNT, GLuint> &mesh, std::vector<T*> &objs);

    template<typename T,typename shader_t, bool shadow>
    void renderLightVolume(IndexedVertexBuffer<VertexNT, GLuint> &mesh, std::vector<T *> &objs, Camera *cam, shader_t *shader);


    void renderPointLights(Camera *cam, bool shadow);
    void renderPointLightsStencil();

    void renderSpotLights(Camera *cam, bool shadow);
    void renderSpotLightsStencil();

    void renderBoxLights(Camera *cam, bool shadow);
    void renderBoxLightsStencil();

    void renderDirectionalLights(Camera *cam, bool shadow);


};

template<typename T>
inline void DeferredLighting::renderStencilVolume(IndexedVertexBuffer<VertexNT, GLuint> &mesh, std::vector<T*> &objs)
{
    setupStencilPass();
    stencilShader->bind();
    stencilShader->uploadView(view);
    stencilShader->uploadProj(proj);
    mesh.bind();
    for(T* &obj : objs){
        if(obj->shouldRender()){

            obj->bindUniformsStencil(*stencilShader);
            mesh.draw();
        }
    }
    mesh.unbind();
    stencilShader->unbind();
}


template<typename T,typename shader_t, bool shadow>
inline void DeferredLighting::renderLightVolume(IndexedVertexBuffer<VertexNT, GLuint> &mesh, std::vector<T*> &objs, Camera *cam, shader_t* shader){

//    SpotLightShader* shader = (shadow)?spotLightShadowShader:spotLightShader;

    shader->bind();
    shader->uploadView(view);
    shader->uploadProj(proj);
    shader->DeferredShader::uploadFramebuffer(&gbuffer);
    shader->uploadScreenSize(vec2(width,height));

    mesh.bind();
    for(T* &obj : objs){
        bool render = (shadow&&obj->shouldCalculateShadowMap()) || (!shadow && obj->shouldRender() && !obj->hasShadows());
        if(render){
            obj->bindUniforms(*shader,cam);
            mesh.draw();
        }
    }
    mesh.unbind();
    shader->unbind();

}


