/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/clusterer.h"

namespace Saiga
{
// TODO: Remove this struct and move the 'refinment' variable to ClustererParameters
// add a comment that it is only used in the cpu clusterer
struct SAIGA_OPENGL_API CPUPlaneClustererParameters : public ClustererParameters
{
    bool refinement = true;

    void fromConfigFile(const std::string& file){};
};

class SAIGA_OPENGL_API CPUPlaneClusterer : public Clusterer
{
   public:
    CPUPlaneClusterer(GLTimerSystem* timer,
                      const ClustererParameters& _params = CPUPlaneClusterer::DefaultParameters());
    CPUPlaneClusterer& operator=(CPUPlaneClusterer& c) = delete;
    ~CPUPlaneClusterer();

    // TODO: This function is broken because it converts CPUPlaneClustererParameters back to ClustererParameters
    static ClustererParameters DefaultParameters()
    {
        CPUPlaneClustererParameters params;
        params.screenSpaceTileSize     = 64;
        params.depthSplits             = 12;
        params.clusterThreeDimensional = true;
        params.useSpecialNearCluster   = true;
        params.specialNearDepthPercent = 0.06f;
        params.refinement              = true;

        return params;
    }

    void renderDebug(Camera* cam) override
    {
        if (clusterDebug) Clusterer::renderDebug(cam);
        if (lightsDebug) lightClustersDebug.render(cam);
    };

    bool refinement = true;

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort) override;

    void clusterLoop(vec3 sphereCenter, float sphereRadius, int index, bool pl, int& itemCount);

    void buildClusters(Camera* cam);


    //
    // Structures for plane arrays.
    //

    std::vector<Plane> planesX;
    std::vector<Plane> planesY;
    std::vector<Plane> planesZ;

    // TODO: Can we get rid of this variable?
    // Just say something like if(itemCount > itemList.size()) itemList.resize(itemList.size() * 2);
    int avgAllowedItemsPerCluster = 128;
    std::vector<std::vector<int32_t>> clusterCache;

    void imgui() override;

    std::vector<Frustum> debugFrusta;
    LineSoup lightClustersDebug;

    bool lightsDebug       = false;
    bool updateLightsDebug = true;


    // TODO: move to ClustererParameters
    bool SAT = false;
};
}  // namespace Saiga
