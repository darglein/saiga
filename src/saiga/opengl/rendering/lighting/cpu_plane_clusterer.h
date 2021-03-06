/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/light_clusterer.h"

namespace Saiga
{
class SAIGA_OPENGL_API CPUPlaneClusterer : public Clusterer
{
   public:
    CPUPlaneClusterer(ClustererParameters _params = ClustererParameters());
    CPUPlaneClusterer& operator=(CPUPlaneClusterer& c) = delete;
    ~CPUPlaneClusterer();

    void clusterLights(Camera* cam, const ViewPort& viewPort) override { clusterLightsInternal(cam, viewPort); }

    void renderDebug(Camera* cam)
    {
        if (!renderDebugEnabled) return;
        if (renderClusterDebug) Clusterer::renderDebug(cam);
        if (renderLightsDebug) gpuDebugFrusta.render(cam);
    };

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort);

    void buildClusters(Camera* cam);


    //
    // Structures for plane arrays.
    //

    std::vector<Plane> planesX;
    std::vector<Plane> planesY;
    std::vector<Plane> planesZ;

    int avgAllowedItemsPerCluster = 128;
    std::vector<std::vector<int>> clusterCache;

    bool refinement = true;

    bool fillImGui() override;

    std::vector<Frustum> debugFrusta;
    LineSoup gpuDebugFrusta;

    bool renderClusterDebug = false;  // TODO Paul: -> UI!
    bool renderLightsDebug  = true;   // TODO Paul: -> UI!

    bool shouldDrawFrustum  = true;
    int debugFrustumX, debugFrustumY, debugFrustumZ;

    mat4 debugCameraModelMatrix;

    bool SAT     = false;
};
}  // namespace Saiga
