/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/clusterer.h"

namespace Saiga
{

class SAIGA_OPENGL_API CPUPlaneClusterer : public Clusterer
{
   public:
    CPUPlaneClusterer(GLTimerSystem* timer,
                      const ClustererParameters& _params = CPUPlaneClusterer::DefaultParameters());
    CPUPlaneClusterer& operator=(CPUPlaneClusterer& c) = delete;
    ~CPUPlaneClusterer();

    static ClustererParameters DefaultParameters()
    {
        ClustererParameters params;
        params.screenSpaceTileSize     = 64;
        params.depthSplits             = 12;
        params.clusterThreeDimensional = true;
        params.useSpecialNearCluster   = true;
        params.specialNearDepthPercent = 0.06f;
        params.refinement              = true;
        params.SAT                     = false;

        return params;
    }

    void renderDebug(Camera* cam) override
    {
        if (clusterDebug) Clusterer::renderDebug(cam);
        if (lightsDebug) lightClustersDebug.render(cam);
    };

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort) override;

    void clusterLoop(vec3 sphereCenter, float sphereRadius, int index, bool pl, int& itemCount);

    void buildClusters(Camera* cam);

    /*
     * ClusterList
     * Gets accessed based on pixel world space position (or screen space on 2D clustering)
     * Looks like this: [offset, plCount, slCount, dlCount], [offset, plCount, slCount, dlCount]
     * ... So for each cluster we store an offset in the itemList and the number of specific lights that were
     * assigned.
     */
    std::vector<Cluster> clusterList;

    /*
     * ItemList
     * Looks like this: [plIdx, slIdx, blIdx, dlIdx], [plIdx, slIdx, blIdx, dlIdx], ...
     * So each item consists of indices for all light types (can be -1, when not set).
     */
    std::vector<ClusterItem> itemList;

    //
    // Structures for plane arrays.
    //

    std::vector<Plane> planesX;
    std::vector<Plane> planesY;
    std::vector<Plane> planesZ;

    std::vector<std::vector<int32_t>> clusterCache;

    void imgui() override;

    std::vector<Frustum> debugFrusta;
    LineSoup lightClustersDebug;

    bool lightsDebug       = false;
    bool updateLightsDebug = true;
};
}  // namespace Saiga
