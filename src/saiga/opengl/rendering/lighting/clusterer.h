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
struct SAIGA_OPENGL_API ClustererParameters
{
    int32_t screenSpaceTileSize = 64;

    bool clusterThreeDimensional = false;

    int32_t depthSplits = 0;

    float specialNearDepthPercent = 0.06f;

    bool useSpecialNearCluster = false;

    /**
     *  Reads all parameters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
     // TODO: remove this function if it's not implemented
    void fromConfigFile(const std::string& file);
};

struct ClusterItem
{
    int32_t lightIndex = 0;
};

struct Cluster
{
    int32_t offset  = -1;
    int32_t plCount = -1;
    int32_t slCount = -1;
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

class SAIGA_OPENGL_API Clusterer
{
   public:
    Clusterer(GLTimerSystem* timer, const ClustererParameters& _params);
    Clusterer& operator=(Clusterer& c) = delete;
    virtual ~Clusterer();

    void resize(int32_t width, int32_t height);

    inline void setParameters(const ClustererParameters& _params)
    {
        params        = _params;
        clustersDirty = true;
    }

    void clusterLights(Camera* cam, const ViewPort& viewPort, ArrayView<PointLight*> pls, ArrayView<SpotLight*> sls);

    virtual void imgui();

    // TODO: virtual? -> wird überschrieben!
    virtual void renderDebug(Camera* cam);

    // TODO: protected?
   public:
    std::vector<PointLightClusterData> pointLightsClusterData;

    std::vector<SpotLightClusterData> spotLightsClusterData;

    Timer lightAssignmentTimer;

   protected:
    ClustererParameters params;
    int32_t width, height;

    int32_t timerIndex = 0;
    double cpuAssignmentTimes[100];

    GLTimerSystem* timer;
    mat4 cached_projection;
    bool clustersDirty = true;

    bool clusterDebug = false;
    bool updateDebug  = false;
    LineSoup debugCluster;
    bool screenSpaceDebug = false;
    bool splitDebug       = false;


    virtual void clusterLightsInternal(Camera* cam, const ViewPort& viewPort){};

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

    // TODO: Struct name!
    struct infoBuf_t
    {
        int32_t clusterX;
        int32_t clusterY;
        int32_t screenSpaceTileSize;
        int32_t screenWidth;
        int32_t screenHeight;
        float zNear;
        float zFar;
        float bias;
        float scale;

        int32_t clusterListCount;
        int32_t itemListCount;
        int32_t tileDebug;
        int32_t splitDebug;

        int32_t specialNearCluster;
        float specialNearDepth;
        int32_t pad2 = 0;
        // TODO: no inline declaration
    } clusterInfoBuffer;

    int32_t getTileIndex(int32_t x, int32_t y, int32_t z)
    {
        return x + clusterInfoBuffer.clusterX * y + (clusterInfoBuffer.clusterX * clusterInfoBuffer.clusterY) * z;
    }


    /*
     * ClusterList
     * Gets accessed based on pixel world space position (or screen space on 2D clustering)
     * Looks like this: [offset, plCount, slCount, dlCount], [offset, plCount, slCount, dlCount]
     * ... So for each cluster we store an offset in the itemList and the number of specific lights that were
     * assigned.
     */
    // TODO: move to cpu clusterer?
    std::vector<Cluster> clusterList;

    /*
     * ItemList
     * Looks like this: [plIdx, slIdx, blIdx, dlIdx], [plIdx, slIdx, blIdx, dlIdx], ...
     * So each item consists of indices for all light types (can be -1, when not set).
     */
    // TODO: move to cpu clusterer?
    std::vector<ClusterItem> itemList;

    // TODO: remove array view of a single element
    // you should be able to pass clusterInfoBuffer direclty to the shader storage buffer update
    ArrayView<infoBuf_t> infoBufferView = {clusterInfoBuffer};

    TemplatedShaderStorageBuffer<infoBuf_t> infoBuffer;
    TemplatedShaderStorageBuffer<Cluster> clusterListBuffer;
    TemplatedShaderStorageBuffer<ClusterItem> itemListBuffer;
};
}  // namespace Saiga
