#pragma once

#include "libhello/opengl/framebuffer.h"
#include "libhello/opengl/mesh.h"

#include "libhello/rendering/lighting/directional_light.h"
#include "libhello/rendering/lighting/point_light.h"
#include "libhello/rendering/lighting/spot_light.h"

class Deferred_Renderer;

class DeferredLighting{
    friend class LightingController;
private:
    int width,height;
    MVPColorShader* debugShader;

    PointLightShader* pointLightShader;
    IndexedVertexBuffer<VertexNT,GLuint> pointLightMesh;
    std::vector<PointLight*> pointLights;

    SpotLightShader* spotLightShader;
    IndexedVertexBuffer<VertexNT,GLuint> spotLightMesh;
    std::vector<SpotLight*> spotLights;

    DirectionalLightShader* directionalLightShader;
    IndexedVertexBuffer<VertexNT,GLuint> directionalLightMesh;
    std::vector<DirectionalLight*> directionalLights;


    MVPShader* stencilShader;
    Framebuffer &framebuffer;

    mat4 inview,view,proj;
    bool drawDebug = true;
public:
    DeferredLighting(Framebuffer &framebuffer);
    ~DeferredLighting();

    void createLightMeshes();
    DirectionalLight* createDirectionalLight();
    PointLight* createPointLight();
    SpotLight* createSpotLight();


    void render(Camera *cam);
    void renderDepthMaps( Deferred_Renderer* renderer );
    void renderDebug();


    void setShader(SpotLightShader* pointLightShader);
    void setShader(PointLightShader* pointLightShader);
    void setShader(DirectionalLightShader* directionalLightShader);

    void setDebugShader(MVPColorShader* shader);
    void setStencilShader(MVPShader* stencilShader);
    void setSize(int width, int height){this->width=width;this->height=height;}

    void setViewProj(const mat4 &iv,const mat4 &v,const mat4 &p);

private:
    void createInputCommands();

    void setupStencilPass();
    void setupLightPass();

    void renderPointLights();
    void renderPointLightsStencil();

    void renderSpotLights();
    void renderSpotLightsStencil();

    void renderDirectionalLights(Camera *cam);

};


