/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/gpu_aabb_clusterer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
#define MIN(X, Y) ((X < Y) ? X : Y)
#define MAX(X, Y) ((X > Y) ? X : Y)
#define MINV(V1, V2) vec3(MIN(V1[0], V2[0]), MIN(V1[1], V2[1]), MIN(V1[2], V2[2]))
#define MAXV(V1, V2) vec3(MAX(V1[0], V2[0]), MAX(V1[1], V2[1]), MAX(V1[2], V2[2]))

GPUAABBClusterer::GPUAABBClusterer(GLTimerSystem* timer, const ClustererParameters& _params) : Clusterer(timer, _params)
{
    lightAssignmentShader = shaderLoader.load<LightAssignmentComputeShader>(assignmentShaderString);
    clusterListBuffer.create(GL_DYNAMIC_DRAW);
    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.create(GL_DYNAMIC_DRAW);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);
}

GPUAABBClusterer::~GPUAABBClusterer() {}

void GPUAABBClusterer::clusterLightsInternal(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    clustersDirty |= cam->proj != cached_projection;
    cached_projection = cam->proj;

    if (clustersDirty) buildClusters(cam);

    {
        auto tim = timer->Measure("Cluster Update");

        int lightCount = lightsClusterData.size();

        if (lightCount > lightClusterDataBuffer.Size())
        {
            lightClusterDataBuffer.create(lightsClusterData, GL_DYNAMIC_DRAW);
            lightClusterDataBuffer.bind(GPU_LIGHT_CLUSTER_DATA_BUFFER_BINDING_POINT);
        }
        else
        {
            lightClusterDataBuffer.update(lightsClusterData, 0);
        }

        clusterInfoBuffer.itemListCount = 0;
        infoBuffer.update(clusterInfoBuffer);
    }

    {
        auto tim = timer->Measure("GPU Light Assignment");

        if(lightAssignmentShader->bind())
        {
            lightAssignmentShader->dispatchCompute(gridCount[0], gridCount[1], gridCount[2]);
            lightAssignmentShader->memoryBarrier(MemoryBarrierMask::GL_BUFFER_UPDATE_BARRIER_BIT);
            lightAssignmentShader->unbind();
        }
    }

    assert_no_glerror();
}

void GPUAABBClusterer::buildClusters(Camera* cam)
{
    clustersDirty = false;
    float camNear = cam->zNear;
    float camFar  = cam->zFar;
    mat4 invProjection(inverse(cam->proj));

    clusterInfoBuffer.screenSpaceTileSize = params.screenSpaceTileSize;
    clusterInfoBuffer.screenWidth         = width;
    clusterInfoBuffer.screenHeight        = height;

    clusterInfoBuffer.zNear = cam->zNear;
    clusterInfoBuffer.zFar  = cam->zFar;

    gridCount[0] = std::ceil((float)width / (float)params.screenSpaceTileSize);
    gridCount[1] = std::ceil((float)height / (float)params.screenSpaceTileSize);
    if (params.clusterThreeDimensional)
        gridCount[2] = params.depthSplits + 1;
    else
        gridCount[2] = 1;

    clusterInfoBuffer.clusterX = (int)gridCount[0];
    clusterInfoBuffer.clusterY = (int)gridCount[1];

    // special near
    float specialNearDepth               = (camFar - camNear) * params.specialNearDepthPercent;
    bool useSpecialNear                  = params.useSpecialNearCluster && specialNearDepth > 0.0f && gridCount[2] > 1;
    clusterInfoBuffer.specialNearCluster = useSpecialNear ? 1 : 0;
    specialNearDepth                     = useSpecialNear ? specialNearDepth : 0.0f;
    clusterInfoBuffer.specialNearDepth   = specialNearDepth;
    float specialGridCount               = gridCount[2] - (useSpecialNear ? 1 : 0);

    clusterInfoBuffer.scale = specialGridCount / log2(camFar / (camNear + specialNearDepth));
    clusterInfoBuffer.bias =
        -(specialGridCount * log2(camNear + specialNearDepth) / log2(camFar / (camNear + specialNearDepth)));

    // Calculate Cluster Planes in View Space.
    int clusterCount = (int)(gridCount[0] * gridCount[1] * gridCount[2]);
    clusterListBuffer.resize(clusterCount);
    clusterInfoBuffer.clusterListCount = clusterCount;

    cullingCluster.clear();
    cullingCluster.resize(clusterCount);
    if (clusterDebug)
    {
        debugCluster.lines.clear();
    }

    // const vec3 eyeView(0.0); Not required because it is zero.

    for (int x = 0; x < (int)gridCount[0]; ++x)
    {
        for (int y = 0; y < (int)gridCount[1]; ++y)
        {
            for (int z = 0; z < (int)gridCount[2]; ++z)
            {
                vec4 screenSpaceBL(x * params.screenSpaceTileSize, y * params.screenSpaceTileSize, -1.0, 1.0);  // Bottom left
                vec4 screenSpaceTR((x + 1) * params.screenSpaceTileSize, (y + 1) * params.screenSpaceTileSize, -1.0,
                                   1.0);  // Top Right

                float tileNear;
                float tileFar;
                if (useSpecialNear && z == 0)
                {
                    tileNear = -camNear;
                    tileFar  = -(camNear + specialNearDepth);
                }
                else
                {
                    int calcZ = useSpecialNear ? z - 1 : z;
                    // Doom Depth Split, because it looks good.
                    tileNear = -(camNear + specialNearDepth) *
                               pow(camFar / (camNear + specialNearDepth), (float)calcZ / specialGridCount);
                    tileFar = -(camNear + specialNearDepth) *
                              pow(camFar / (camNear + specialNearDepth), (float)(calcZ + 1) / specialGridCount);
                }

                vec3 viewNearPlaneBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
                vec3 viewNearPlaneTR(make_vec3(viewPosFromScreenPos(screenSpaceTR, invProjection)));

                vec3 viewNearClusterBL(zeroZIntersection(viewNearPlaneBL, tileNear));
                vec3 viewNearClusterTR(zeroZIntersection(viewNearPlaneTR, tileNear));

                vec3 viewFarClusterBL(zeroZIntersection(viewNearPlaneBL, tileFar));
                vec3 viewFarClusterTR(zeroZIntersection(viewNearPlaneTR, tileFar));

                vec3 minBL = MINV(viewNearClusterBL, viewFarClusterBL);
                vec3 minTR = MINV(viewNearClusterTR, viewFarClusterTR);
                vec3 maxBL = MAXV(viewNearClusterBL, viewFarClusterBL);
                vec3 maxTR = MAXV(viewNearClusterTR, viewFarClusterTR);

                vec3 AABBmin = MINV(minBL, minTR);
                vec3 AABBmax = MAXV(maxBL, maxTR);

                int tileIndex = x + (int)gridCount[0] * y + (int)(gridCount[0] * gridCount[1]) * z;

                cullingCluster.at(tileIndex).center  = (AABBmin + AABBmax) * 0.5f;
                cullingCluster.at(tileIndex).extends = AABBmax - cullingCluster.at(tileIndex).center;

                const AABB box(AABBmin, AABBmax);

                if (clusterDebug)
                {
                    PointVertex v;

                    v.color = vec3(1.0, 0.75, 0);

                    v.position = box.cornerPoint(0);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(3);
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(7);
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(4);
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(0);
                    debugCluster.lines.push_back(v);

                    v.position = box.cornerPoint(1);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(2);
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(6);
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(5);
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(1);
                    debugCluster.lines.push_back(v);

                    v.position = box.cornerPoint(0);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(1);
                    debugCluster.lines.push_back(v);

                    v.position = box.cornerPoint(3);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(2);
                    debugCluster.lines.push_back(v);

                    v.position = box.cornerPoint(4);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(5);
                    debugCluster.lines.push_back(v);

                    v.position = box.cornerPoint(7);
                    debugCluster.lines.push_back(v);
                    v.position = box.cornerPoint(6);
                    debugCluster.lines.push_back(v);
                }
            }
        }
    }

    if (clusterDebug)
    {
        debugCluster.lineWidth = 2;

        debugCluster.setModelMatrix(cam->getModelMatrix());  // is inverse view.
        debugCluster.translateLocal(vec3(0, 0, -0.0001f));
#if 0
        debugCluster.setPosition(make_vec4(0));
        debugCluster.translateGlobal(vec3(0, 6, 0));
        debugCluster.setScale(make_vec3(0.33f));
#endif
        debugCluster.calculateModel();
        debugCluster.updateBuffer();
        updateDebug = false;
    }

    {
        auto tim                     = timer->Measure("Info And Structure Update");
        clusterInfoBuffer.tileDebug  = screenSpaceDebug ? 1024 : 0; // 1024 is a good size for the compute shader shared arrays.
        clusterInfoBuffer.splitDebug = splitDebug ? 1 : 0;

        itemListBuffer.resize(1024 * clusterInfoBuffer.clusterListCount); // 1024 is a good size for the compute shader shared arrays.
        clusterInfoBuffer.itemListCount = 0;

        infoBuffer.update(clusterInfoBuffer);

        if (cullingCluster.size() > clusterStructuresBuffer.Size())
        {
            clusterStructuresBuffer.create(cullingCluster, GL_DYNAMIC_DRAW);

            clusterStructuresBuffer.bind(GPU_LIGHT_CLUSTER_CLUSTER_STRUCTURES_BINDING_POINT);
        }
        else
        {
            clusterStructuresBuffer.update(cullingCluster);
        }
    }
}

}  // namespace Saiga
