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

class Deferred_Renderer;

class DeferredLighting{
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


    MVPShader* stencilShader;
    Framebuffer &framebuffer;


    mat4 inview,view,proj;
    bool drawDebug = true;

//    raw_Texture* dummyTexture;
//    raw_Texture* dummyCubeTexture;


public:
    int totalLights;
    int visibleLights;
    int renderedDepthmaps;

    Texture* ssaoTexture;

    DeferredLighting(Framebuffer &framebuffer);
    ~DeferredLighting();

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
    void renderDepthMaps(RendererInterface *renderer );
    void renderDebug();


    void setShader(SpotLightShader* spotLightShader, SpotLightShader* spotLightShadowShader);
    void setShader(PointLightShader* pointLightShader,PointLightShader* pointLightShadowShader);
    void setShader(DirectionalLightShader* directionalLightShader,DirectionalLightShader* directionalLightShadowShader);
    void setShader(BoxLightShader* boxLightShader,BoxLightShader* boxLightShadowShader);

    void setDebugShader(MVPColorShader* shader);
    void setStencilShader(MVPShader* stencilShader);
    void setSize(int width, int height){this->width=width;this->height=height;}

    void setViewProj(const mat4 &iv,const mat4 &v,const mat4 &p);

    void cullLights(Camera *cam);
private:
    void createInputCommands();

    void setupStencilPass();
    void setupLightPass();

    void renderPointLights(Camera *cam, bool shadow);
    void renderPointLightsStencil();

    void renderSpotLights(Camera *cam, bool shadow);
    void renderSpotLightsStencil();

    void renderBoxLights(Camera *cam, bool shadow);
    void renderBoxLightsStencil();

    void renderDirectionalLights(Camera *cam, bool shadow);

};


