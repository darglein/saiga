/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/clusterer.h"

namespace Saiga
{

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

class SAIGA_OPENGL_API GPUAABBClusterer : public Clusterer
{
   public:
    GPUAABBClusterer(GLTimerSystem* timer, const ClustererParameters& _params = GPUAABBClusterer::DefaultParameters());
    GPUAABBClusterer& operator=(GPUAABBClusterer& c) = delete;
    ~GPUAABBClusterer();

    static ClustererParameters DefaultParameters()
    {
        ClustererParameters params;
        params.screenSpaceTileSize = 64;
        params.depthSplits = 24;
        params.clusterThreeDimensional = true;
        params.useSpecialNearCluster = true;
        params.specialNearDepthPercent = 0.06f;

        return params;
    }

   private:
    void clusterLightsInternal(Camera* cam, const ViewPort& viewPort) override;

    void buildClusters(Camera* cam);

    //
    // Structures for AABB cluster boundaries.
    //

    struct ClusterBounds
    {
        vec3 center;
        float pad0;
        vec3 extends;
        float pad1;
    };

    std::vector<ClusterBounds> cullingCluster;

    TemplatedShaderStorageBuffer<LightBoundingSphere> lightClusterDataBuffer;

    TemplatedShaderStorageBuffer<ClusterBounds> clusterStructuresBuffer;

    float gridCount[3];

    std::shared_ptr<LightAssignmentComputeShader> lightAssignmentShader;

    const char* assignmentShaderString = "lighting/light_assignment.glsl";
};
}  // namespace Saiga
