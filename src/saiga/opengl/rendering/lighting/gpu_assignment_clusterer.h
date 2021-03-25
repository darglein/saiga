/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/light_clusterer.h"

namespace Saiga
{
class SAIGA_OPENGL_API GPUAssignmentClusterer : public Clusterer
{
   public:
    GPUAssignmentClusterer(GLTimerSystem* timer, ClustererParameters _params = ClustererParameters());
    GPUAssignmentClusterer& operator=(GPUAssignmentClusterer& c) = delete;
    ~GPUAssignmentClusterer();

    void clusterLights(Camera* cam, const ViewPort& viewPort) override { clusterLightsInternal(cam, viewPort); }

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort);

    void buildClusters(Camera* cam);

    //_d
    // Structures for AABB cluster boundaries.
    //

    struct clusterBounds
    {
        vec3 minB;
        float pad0;
        vec3 maxB;
        float pad1;
    };

    int allowedItemsPerCluster = 1024;
    std::vector<clusterBounds> cullingCluster;

    int allowedLights = 66000;
    ShaderStorageBuffer lightClusterDataBuffer;
    ShaderStorageBuffer clusterStructuresBuffer;

    float gridCount[3];

    std::shared_ptr<LightAssignmentComputeShader> lightAssignmentShader;

    const char* assignmentShaderString = "lighting/light_assignment.glsl";
};
}  // namespace Saiga
