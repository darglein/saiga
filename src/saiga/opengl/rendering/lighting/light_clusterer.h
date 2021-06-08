/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/intersection.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/shaderStorageBuffer.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/pointCloud.h"

namespace Saiga
{

// TODO: classes start with a capital letter
struct clusterItem
{
    int32_t lightIndex = 0;
};
struct cluster
{
    // TODO: inconsistent int32_t <-> int
    int offset  = -1;
    int plCount = -1;
    int slCount = -1;
};

struct PointLightClusterData
{
    vec3 world_center;
    float radius;
    PointLightClusterData(vec3 w_center, float r) : world_center(w_center), radius(r) {}
};

struct SpotLightClusterData
{
    vec3 world_center;  // should be sufficient -> center position of the spot light cone
    float radius;       // should be sufficient -> bounding sphere instead of transformed cone
    SpotLightClusterData(vec3 w_center, float r) : world_center(w_center), radius(r) {}
};

// TODO: move to gpu impl
class SAIGA_OPENGL_API LightAssignmentComputeShader : public Shader
{
   public:
    GLint location_lightInfoBlock;

    virtual void checkUniforms() override
    {
        Shader::checkUniforms();

        location_lightInfoBlock = getUniformBlockLocation("lightInfoBlock");
        setUniformBlockBinding(location_lightInfoBlock, LIGHT_INFO_BINDING_POINT);
    }
};

// TODO: should match file name either rename file or rename this class
class SAIGA_OPENGL_API Clusterer
{
   public:
    Clusterer(GLTimerSystem* timer);
    Clusterer& operator=(Clusterer& c) = delete;
    virtual ~Clusterer();

    // TODO: remove init? resize should be enough
    void init(int width, int height);
    void resize(int width, int height);

    // TODO: create a parameter struct. see renderer.h for an example
    inline void enable3DClusters(bool enabled)
    {
        clusterThreeDimensional = enabled;
        clustersDirty           = true;
    }

    inline void set(int _tileSize, int _depthSplits)
    {
        depthSplits         = _depthSplits;
        screenSpaceTileSize = _tileSize;
        clustersDirty       = true;
    }

    // TODO: unused?
    inline bool clusters3D() { return clusterThreeDimensional; }

    // TODO: unused?
    void loadComputeShaders();

    // TODO: change interface. instead of the 4 functions below can we make it to a single:
    //  virtual void clusterLights(Camera* cam, const ViewPort& viewPort, ArrayView<PointLight*> pls,
    //                          ArrayView<SpotLight*> sls)
    // See shadow_manager.h for an example
    inline void clearLightData()
    {
        pointLightsClusterData.clear();
        spotLightsClusterData.clear();
    }

    inline void addPointLight(const vec3& position, const float& radius)
    {
        pointLightsClusterData.emplace_back(position, radius);
    }

    inline void addSpotLight(const vec3& position, const float& radius)
    {
        spotLightsClusterData.emplace_back(position, radius);
    }

    // Binds Cluster and Item ShaderStorageBuffers at the end.
    virtual void clusterLights(Camera* cam, const ViewPort& viewPort) = 0;



    virtual void imgui();

    // TODO: virtual? + move to .cpp
    virtual void renderDebug(Camera* cam)
    {
        if (!clusterDebug) return;
        debugCluster.render(cam);
    };

   public:
    std::vector<PointLightClusterData> pointLightsClusterData;

    std::vector<SpotLightClusterData> spotLightsClusterData;

    Timer lightAssignmentTimer;

   protected:
    int width, height;

    int timerIndex          = 0;
    int screenSpaceTileSize = 128;
    int depthSplits         = 0;
    double cpuAssignmentTimes[100];

    GLTimerSystem* timer;
    mat4 cached_projection;
    bool clustersDirty = true;

    bool clusterThreeDimensional = false;

    bool clusterDebug = false;
    bool updateDebug  = false;
    LineSoup debugCluster;
    bool screenSpaceDebug = false;
    bool splitDebug       = false;

    float specialNearDepthPercent = 0.06f;
    bool useSpecialNearCluster    = true;


    vec4 viewPosFromScreenPos(vec4 screen, const mat4& inverseProjection)
    {
        // to ndc
        vec2 ndc(clamp(screen.x() / width, 0.0, 1.0), clamp(screen.y() / height, 0.0, 1.0));

        // to clip
        vec4 clip(ndc.x() * 2.0f - 1.0f, ndc.y() * 2.0f - 1.0f, screen.z(), screen.w());

        // to view
        vec4 view(inverseProjection * clip);
        view /= view.w();

        return view;
    }

    // for eye = vec3(0) this is unoptimized.
    vec3 eyeZIntersection(vec3 eye, vec3 through, float z)
    {
        vec3 viewSpacePlaneOrientation(0.0f, 0.0f, 1.0f);

        vec3 line(through - eye);

        float t = (z - viewSpacePlaneOrientation.dot(eye)) / viewSpacePlaneOrientation.dot(line);

        return eye + t * line;
    }

    vec3 zeroZIntersection(vec3 through, float z) { return through * z / through.z(); }

    struct infoBuf_t
    {
        int clusterX;
        int clusterY;
        int screenSpaceTileSize;
        int screenWidth;
        int screenHeight;
        float zNear;
        float zFar;
        float bias;
        float scale;

        int clusterListCount;
        int itemListCount;
        int tileDebug;
        int splitDebug;

        int specialNearCluster;
        float specialNearDepth;
        int pad2 = 0;
    } clusterInfoBuffer;

    int getTileIndex(int x, int y, int z)
    {
        return x + clusterInfoBuffer.clusterX * y + (clusterInfoBuffer.clusterX * clusterInfoBuffer.clusterY) * z;
    }

    // TODO: this struct and the struct below seem unnecessary
    struct clusterBuffer_t
    {
        std::vector<cluster> clusterList;
        /*
         * ClusterList
         * Gets accessed based on pixel world space position (or screen space on 2D clustering)
         * Looks like this: [offset, plCount, slCount, dlCount], [offset, plCount, slCount, dlCount]
         * ... So for each cluster we store an offset in the itemList and the number of specific lights that were
         * assigned.
         */
    } clusterBuffer;

    struct itemBuffer_t
    {
        std::vector<clusterItem> itemList;
        /*
         * ItemList
         * Looks like this: [plIdx, slIdx, blIdx, dlIdx], [plIdx, slIdx, blIdx, dlIdx], ...
         * So each item consists of indices for all light types (can be -1, when not set).
         */
    } itemBuffer;

    // TODO: use TemplatedShaderStorageBuffer
    ShaderStorageBuffer infoBuffer;
    ShaderStorageBuffer clusterListBuffer;
    ShaderStorageBuffer itemListBuffer;
};
}  // namespace Saiga
