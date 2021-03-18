/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/light_clusterer.h"

namespace Saiga
{
class SAIGA_OPENGL_API SixPlaneClusterer : public Clusterer
{
   public:
    SixPlaneClusterer(GLTimerSystem* timer, ClustererParameters _params = ClustererParameters());
    SixPlaneClusterer& operator=(SixPlaneClusterer& c) = delete;
    ~SixPlaneClusterer();

    void clusterLights(Camera* cam, const ViewPort& viewPort) override { clusterLightsInternal(cam, viewPort); }

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort);

    void buildClusters(Camera* cam);

    //
    // Structures for 6 plane cluster boundaries.
    //

    struct cluster_bounds
    {
        std::array<Plane, 6> planes;
    };

    std::vector<cluster_bounds> culling_cluster;
};
}  // namespace Saiga
