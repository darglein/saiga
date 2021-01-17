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
#include "saiga/opengl/rendering/lighting/renderer_lighting.h"
#include "saiga/opengl/shader/basic_shaders.h"

namespace Saiga
{
using LightID   = int32_t;
using ClusterID = int32_t;

struct PointLightClusterData
{
    vec3 world_center;
    float radius;
};

struct SpotLightClusterData
{
    vec3 world_center;  // should be sufficient -> center position of the spot light cone
    float radius;       // should be sufficient -> bounding sphere instead of transformed cone
};

struct BoxLightClusterData
{
    vec3 world_center;
    float radius;  // should be sufficient -> bounding sphere instead of transformed box
};

class SAIGA_OPENGL_API LightToClusterComputeShader : public Shader
{
   public:
    GLint location_clusterDataBlockPointLights;  // pointLightClusterData array
    GLint location_clusterDataBlockSpotLights;   // spotLightClusterData array
    GLint location_clusterDataBlockBoxLights;    // boxLightClusterData array
    GLint location_clusterInfoBlock;             // number of lights in cluster arrays
    GLint location_clusters;                     // clusters

    GLint location_lightToClusterMap;  // light to cluster map

    virtual void checkUniforms() override{};
};

class SAIGA_OPENGL_API BuildClusterComputeShader : public Shader
{
   public:
    GLint location_viewFrustumData;  // width, height, depth, splitX, splitY, splitZ

    GLint location_clusters;  // clusters

    virtual void checkUniforms() override{};
};

struct SAIGA_OPENGL_API ClustererParameters
{
    bool clusterThreeDimensional = false;
    bool useTimers               = true;

    void fromConfigFile(const std::string& file) {}
};

class SAIGA_OPENGL_API Clusterer
{
   public:
    // vars

    Clusterer(ClustererParameters _params = ClustererParameters()){};
    Clusterer& operator=(Clusterer& c) = delete;
    ~Clusterer(){};

    void init(int width, int height, bool _useTimers){};
    void resize(int width, int height){};

    inline void enable3DClusters(bool enabled) { clusterThreeDimensional = enabled; }

    inline bool clusters3D() { return clusterThreeDimensional; }

    void loadComputeShaders(){};

    inline std::vector<PointLightClusterData>& pointLightClusterData()
    {
        pointLightsClusterData.clear();
        return pointLightsClusterData;
    }

    inline std::vector<SpotLightClusterData>& spotLightClusterData()
    {
        spotLightsClusterData.clear();
        return spotLightsClusterData;
    }

    inline std::vector<BoxLightClusterData>& boxLightClusterData()
    {
        boxLightsClusterData.clear();
        return boxLightsClusterData;
    }

    void clusterLights(Camera* cam, const ViewPort& viewPort){};

    const std::unordered_map<LightID, ClusterID>& getLightToClusterMap(bool& dirty)
    {
        dirty = clustersDirty;
        return lightToClusterMap;
    }

    void setDebugShader(std::shared_ptr<Shader> shader){};

    void printTimings(){};
    void renderImGui(bool* p_open = NULL){};
    void renderDebug(bool* p_open = NULL){};

   public:
    int width, height;

    // the vertex position is sufficient. no normals and texture coordinates needed.
    typedef IndexedVertexBuffer<Vertex, GLushort> lightMesh_t;

    lightMesh_t pointLightMesh;
    std::vector<PointLightClusterData> pointLightsClusterData;

    lightMesh_t spotLightMesh;
    std::vector<SpotLightClusterData> spotLightsClusterData;

    lightMesh_t boxLightMesh;
    std::vector<BoxLightClusterData> boxLightsClusterData;

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

   private:
    bool clusterThreeDimensional = false;
    bool useTimers;

    std::unordered_map<LightID, ClusterID> lightToClusterMap;
    bool clustersDirty = true;


    std::shared_ptr<MVPColorShader> debugShader;
    std::shared_ptr<BuildClusterComputeShader> buildClusterShader2D;
    std::shared_ptr<BuildClusterComputeShader> buildClusterShader3D;
    std::shared_ptr<LightToClusterComputeShader> buildLightToClusterMapShader;
};
}  // namespace Saiga
