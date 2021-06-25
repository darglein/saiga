/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/clusterer.h"

namespace Saiga
{
class SAIGA_OPENGL_API SixPlaneClusterer : public Clusterer
{
   public:
    SixPlaneClusterer(GLTimerSystem* timer, const ClustererParameters& _params = SixPlaneClusterer::DefaultParameters());
    SixPlaneClusterer& operator=(SixPlaneClusterer& c) = delete;
    ~SixPlaneClusterer();

    static ClustererParameters DefaultParameters()
    {
        ClustererParameters params;
        params.screenSpaceTileSize = 128;
        params.depthSplits = 0;
        params.clusterThreeDimensional = false;
        params.useSpecialNearCluster = false;

        return params;
    }

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort) override;

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
    // Structures for 6 plane cluster boundaries.
    //

    struct ClusterBounds
    {
        std::array<Plane, 6> planes;
    };

    std::vector<std::vector<int32_t>> clusterCache;
    std::vector<ClusterBounds> cullingCluster;
};
}  // namespace Saiga
