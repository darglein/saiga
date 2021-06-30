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

#include "light_manager.h"
#include "shadow_manager.h"

namespace Saiga
{
class PointLightShader;
class SpotLightShader;
class DirectionalLightShader;
class LightAccumulationShader;
class GLTimerSystem;

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

#define GPU_LIGHT_CLUSTER_DATA_BUFFER_BINDING_POINT 10
#define GPU_LIGHT_CLUSTER_CLUSTER_STRUCTURES_BINDING_POINT 11

class Clusterer;
class SAIGA_OPENGL_API RendererLighting : public LightManager
{
   public:
    vec4 clearColor = make_vec4(0);


    int shadowSamples = 16;  // Quadratic number (1,4,9,16,...)

    bool drawDebug = false;



    RendererLighting(GLTimerSystem* timer);
    RendererLighting& operator=(RendererLighting& l) = delete;
    ~RendererLighting();

    virtual void init(int width, int height, bool _useTimers);
    virtual void resize(int width, int height);

    virtual void loadShaders();
    void createLightMeshes();

    virtual void initRender();
    virtual void render(Camera* cam, const ViewPort& viewPort);
    virtual void renderDepthMaps(Camera* camera, RenderingInterface* renderer);
    virtual void renderDebug(Camera* cam);

    void setDebugShader(std::shared_ptr<MVPColorShader> shader);

    // Compute culling and the light statistics (see variables below)
    void ComputeCullingAndStatistics(Camera* cam);

    virtual void renderImGui();

    virtual void setClusterType(int tp){};

    virtual std::shared_ptr<Clusterer> getClusterer() { return nullptr; };


   public:
    int width, height;
    std::shared_ptr<MVPColorShader> debugShader;

    // the vertex position is sufficient. no normals and texture coordinates needed.
    typedef IndexedVertexBuffer<Vertex, uint32_t> lightMesh_t;

    std::shared_ptr<PointLightShader> pointLightShader, pointLightShadowShader;
    lightMesh_t pointLightMesh;

    std::shared_ptr<SpotLightShader> spotLightShader, spotLightShadowShader;
    lightMesh_t spotLightMesh;
    std::shared_ptr<DirectionalLightShader> directionalLightShader, directionalLightShadowShader;

    lightMesh_t directionalLightMesh;

    ShaderPart::ShaderCodeInjections shadowInjection;

    bool lightDepthTest = true;


   protected:
    GLTimerSystem* timer;

    ShadowManager shadowManager;

    TemplatedShaderStorageBuffer<PointLight::ShaderData> point_light_data;
    TemplatedShaderStorageBuffer<SpotLight::ShaderData> spot_light_data;
    TemplatedShaderStorageBuffer<DirectionalLight::ShaderData> directional_light_data;
};
}  // namespace Saiga
