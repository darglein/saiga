/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/lighting/directional_light.h"
#include "saiga/opengl/rendering/lighting/point_light.h"
#include "saiga/opengl/rendering/lighting/spot_light.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/uniformBuffer.h"
#include "saiga/opengl/vertex.h"

#include <set>

namespace Saiga
{
class PointLightShader;
class SpotLightShader;
class DirectionalLightShader;
class LightAccumulationShader;


struct RendererLightingShaderNames
{
    std::string pointLightShader       = "lighting/light_point.glsl";
    std::string spotLightShader        = "lighting/light_spot.glsl";
    std::string directionalLightShader = "lighting/light_directional.glsl";
    std::string debugShader            = "lighting/debugmesh.glsl";
    std::string stencilShader          = "lighting/stenciltest.glsl";
    std::string lightingUberShader     = "lighting/lighting_uber.glsl";
};

namespace uber
{

struct LightData
{
    std::vector<PointLight::ShaderData> pointLights;
    std::vector<SpotLight::ShaderData> spotLights;
    std::vector<DirectionalLight::ShaderData> directionalLights;
};

struct LightInfo
{
    int pointLightCount;
    int spotLightCount;
    int directionalLightCount;

    int clusterEnabled;
};
}  // namespace uber

#define POINT_LIGHT_DATA_BINDING_POINT 2
#define SPOT_LIGHT_DATA_BINDING_POINT 3
#define BOX_LIGHT_DATA_BINDING_POINT 4
#define DIRECTIONAL_LIGHT_DATA_BINDING_POINT 5
#define LIGHT_INFO_BINDING_POINT 6

#define LIGHT_CLUSTER_INFO_BINDING_POINT 7
#define LIGHT_CLUSTER_LIST_BINDING_POINT 8
#define LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT 9

class SAIGA_OPENGL_API RendererLighting
{
   public:
    vec4 clearColor = make_vec4(0);
    int totalLights;
    int visibleLights;
    int renderedDepthmaps;

    int shadowSamples = 16;  // Quadratic number (1,4,9,16,...)

    bool drawDebug = false;

    bool useTimers = true;

    bool backFaceShadows     = false;
    float shadowOffsetFactor = 2;
    float shadowOffsetUnits  = 10;

    RendererLighting();
    RendererLighting& operator=(RendererLighting& l) = delete;
    ~RendererLighting();

    virtual void init(int width, int height, bool _useTimers);
    virtual void resize(int width, int height);

    virtual void loadShaders();
    void createLightMeshes();

    void setRenderDebug(bool b) { drawDebug = b; }

    void AddLight(std::shared_ptr<DirectionalLight> l) { directionalLights.insert(l); }
    void AddLight(std::shared_ptr<PointLight> l) { pointLights.insert(l); }
    void AddLight(std::shared_ptr<SpotLight> l) { spotLights.insert(l); }

    void removeLight(std::shared_ptr<DirectionalLight> l) { directionalLights.erase(l); }
    void removeLight(std::shared_ptr<PointLight> l) { pointLights.erase(l); }
    void removeLight(std::shared_ptr<SpotLight> l) { spotLights.erase(l); }

    void setShader(std::shared_ptr<SpotLightShader> spotLightShader,
                   std::shared_ptr<SpotLightShader> spotLightShadowShader);
    void setShader(std::shared_ptr<PointLightShader> pointLightShader,
                   std::shared_ptr<PointLightShader> pointLightShadowShader);
    void setShader(std::shared_ptr<DirectionalLightShader> directionalLightShader,
                   std::shared_ptr<DirectionalLightShader> directionalLightShadowShader);

    virtual void initRender();
    virtual void render(Camera* cam, const ViewPort& viewPort);
    virtual void renderDepthMaps(RenderingInterface* renderer);
    virtual void renderDebug(Camera* cam);

    void setDebugShader(std::shared_ptr<MVPColorShader> shader);

    virtual void cullLights(Camera* cam);

    void printTimings();
    virtual void renderImGui();

    virtual void setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights);


   public:
    int width, height;
    std::shared_ptr<MVPColorShader> debugShader;
    UniformBuffer shadowCameraBuffer;

    // the vertex position is sufficient. no normals and texture coordinates needed.
    typedef IndexedVertexBuffer<Vertex, uint32_t> lightMesh_t;

    std::shared_ptr<PointLightShader> pointLightShader, pointLightShadowShader;
    lightMesh_t pointLightMesh;
    std::set<std::shared_ptr<PointLight> > pointLights;

    std::shared_ptr<SpotLightShader> spotLightShader, spotLightShadowShader;
    lightMesh_t spotLightMesh;
    std::set<std::shared_ptr<SpotLight> > spotLights;

    std::shared_ptr<DirectionalLightShader> directionalLightShader, directionalLightShadowShader;
    lightMesh_t directionalLightMesh;
    std::set<std::shared_ptr<DirectionalLight> > directionalLights;

    ShaderPart::ShaderCodeInjections shadowInjection;

    bool lightDepthTest = true;

    std::vector<FilteredMultiFrameOpenGLTimer> timers2;
    std::vector<std::string> timerStrings;
    void startTimer(int timer)
    {
        if (useTimers) timers2[timer].startTimer();
    }
    void stopTimer(int timer)
    {
        if (useTimers) timers2[timer].stopTimer();
    }
    float getTime(int timer)
    {
        if (!useTimers) return 0;
        return timers2[timer].getTimeMS();
    }

   protected:
    int maximumNumberOfDirectionalLights = 256;
    int maximumNumberOfPointLights       = 256;
    int maximumNumberOfSpotLights        = 256;

    bool showLightingImgui = false;
    int selected_light     = -1;
    int selecte_light_type = 0;
    std::shared_ptr<LightBase> selected_light_ptr;
};
}  // namespace Saiga
