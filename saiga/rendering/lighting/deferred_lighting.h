#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/camera/camera.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/query/gpuTimer.h"

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

    UniformBuffer shadowCameraBuffer;

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

//    MVPTextureShader* blitDepthShader;
    LightAccumulationShader* lightAccumulationShader;

    MVPShader* stencilShader;
    GBuffer &gbuffer;


    bool drawDebug = true;

	bool useTimers = true;


    std::vector<FilteredGPUTimer> timers2;
	std::vector<std::string> timerStrings;
    void startTimer(int timer){if(useTimers)timers2[timer].startTimer();}
    void stopTimer(int timer){if(useTimers)timers2[timer].stopTimer();}
	float getTime(int timer) { if (!useTimers) return 0; return timers2[timer].getTimeMS(); }
public:
    int totalLights;
    int visibleLights;
    int renderedDepthmaps;
    int currentStencilId = 0;

    int shadowSamples = 16; //Quadratic number (1,4,9,16,...)

    Texture* ssaoTexture;

    Texture* lightAccumulationTexture;
    Framebuffer lightAccumulationBuffer;

    DeferredLighting(GBuffer &gbuffer);
	DeferredLighting& operator=(DeferredLighting& l) = delete;
    ~DeferredLighting();

    void init(int width, int height, bool _useTimers);
    void resize(int width, int height);

    void loadShaders();

    void setRenderDebug(bool b){drawDebug = b;}
    void createLightMeshes();

    DirectionalLight* createDirectionalLight();
    PointLight* createPointLight();
    SpotLight* createSpotLight();
    BoxLight* createBoxLight();

    void removeLight(DirectionalLight* l);
    void removeLight(PointLight* l);
    void removeLight(SpotLight* l);
    void removeLight(BoxLight* l);


    void render(Camera *cam);
    void renderLightAccumulation();
    void renderDepthMaps(Program *renderer );
    void renderDebug(Camera *cam);


    void setShader(SpotLightShader* spotLightShader, SpotLightShader* spotLightShadowShader);
    void setShader(PointLightShader* pointLightShader,PointLightShader* pointLightShadowShader);
    void setShader(DirectionalLightShader* directionalLightShader,DirectionalLightShader* directionalLightShadowShader);
    void setShader(BoxLightShader* boxLightShader,BoxLightShader* boxLightShadowShader);

    void setDebugShader(MVPColorShader* shader);
    void setStencilShader(MVPShader* stencilShader);



    void cullLights(Camera *cam);

    void printTimings();

private:

    void blitGbufferDepthToAccumulationBuffer();
    void setupStencilPass();
    void setupLightPass();

    template<typename T,typename shader_t>
    void renderLightVolume(lightMesh_t &mesh, T* obj, Camera *cam, shader_t *shader , shader_t *shaderShadow);


    void renderDirectionalLights(Camera *cam, bool shadow);

};


template<typename T,typename shader_t>
inline void DeferredLighting::renderLightVolume(lightMesh_t &mesh, T* obj, Camera *cam, shader_t *shaderNormal , shader_t *shaderShadow){
    if(!obj->shouldRender())
        return;

    setupStencilPass();
    stencilShader->bind();

    obj->bindUniformsStencil(*stencilShader);
    mesh.bindAndDraw();
    stencilShader->unbind();


    setupLightPass();
    shader_t* shader = (obj->hasShadows()) ? shaderShadow : shaderNormal;
    shader->bind();
    shader->DeferredShader::uploadFramebuffer(&gbuffer);
    shader->uploadScreenSize(vec2(width,height));

    obj->bindUniforms(*shader,cam);
    mesh.bindAndDraw();
    shader->unbind();


}
