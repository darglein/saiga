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
    // The size of the screen space tiles. Tiles are quadratic.
    int32_t screenSpaceTileSize = 64;

    // True if the camera frustum should also be split in the camera z direction.
    bool clusterThreeDimensional = false;

    // The number of depth splits.
    int32_t depthSplits = 0;

    // True if a special first cluster should be used.
    bool useSpecialNearCluster = false;

    // The distance from the near plane to the first split, when useSpecialNearCluster is true.
    // 0.0 means split == camera near plane depth, 1.0 means split == camera far plane depth.
    float specialNearDepthPercent = 0.06f;

    // Extra refinement for better assignment. Is only used in the CPUPlaneClusterer.
    bool refinement = true;

    // Debug Seperating Axis Theorem. Attention: SLOW. Is only used in the CPUPlaneClusterer.
    bool SAT = false;
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

struct LightBoundingSphere
{
    vec3 world_center;
    float radius;
    LightBoundingSphere(vec3 w_center, float r) : world_center(w_center), radius(r) {}
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

   protected:
    std::vector<LightBoundingSphere> lightsClusterData;
    int32_t pointLightCount;

    Timer lightAssignmentTimer;
    GLTimerSystem* timer;
    ClustererParameters params;
    int32_t width, height;

    int32_t timerIndex = 0;
    double cpuAssignmentTimes[100];

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

    struct ClustererInfoBuffer
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
    };

    ClustererInfoBuffer clusterInfoBuffer;

    int32_t getTileIndex(int32_t x, int32_t y, int32_t z)
    {
        return x + clusterInfoBuffer.clusterX * y + (clusterInfoBuffer.clusterX * clusterInfoBuffer.clusterY) * z;
    }

    TemplatedShaderStorageBuffer<ClustererInfoBuffer> infoBuffer;
    TemplatedShaderStorageBuffer<Cluster> clusterListBuffer;
    TemplatedShaderStorageBuffer<ClusterItem> itemListBuffer;
};
}  // namespace Saiga
