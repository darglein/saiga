/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/gpu_assignment_clusterer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/shader/shaderLoader.h"

namespace Saiga
{
#define MIN(X, Y) ((X < Y) ? X : Y)
#define MAX(X, Y) ((X > Y) ? X : Y)
#define MINV(V1, V2) vec3(MIN(V1[0], V2[0]), MIN(V1[1], V2[1]), MIN(V1[2], V2[2]))
#define MAXV(V1, V2) vec3(MAX(V1[0], V2[0]), MAX(V1[1], V2[1]), MAX(V1[2], V2[2]))

GPUAssignmentClusterer::GPUAssignmentClusterer(GLTimerSystem* timer, ClustererParameters _params)
    : Clusterer(timer, _params)
{
    lightAssignmentShader = shaderLoader.load<LightAssignmentComputeShader>(assignmentShaderString);
    lightClusterDataBuffer.createGLBuffer(nullptr, allowedLights * sizeof(PointLightClusterData), GL_DYNAMIC_DRAW);
    clusterStructuresBuffer.createGLBuffer(nullptr, 0, GL_DYNAMIC_DRAW);
}

GPUAssignmentClusterer::~GPUAssignmentClusterer() {}

void GPUAssignmentClusterer::clusterLightsInternal(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    clustersDirty |= cam->proj != cached_projection;
    cached_projection = cam->proj;

    if (clustersDirty) buildClusters(cam);

    {
        auto tim = timer->CreateScope("Cluster Update");

        int plSize = sizeof(PointLightClusterData) * pointLightsClusterData.size();
        lightClusterDataBuffer.updateBuffer(pointLightsClusterData.data(), plSize, 0);
        int slSize = sizeof(SpotLightClusterData) * spotLightsClusterData.size();
        lightClusterDataBuffer.updateBuffer(spotLightsClusterData.data(), slSize, plSize);

        clusterInfoBuffer.itemListCount = 0;
        infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);

        // TODO Paul: Hardcoded!
        lightClusterDataBuffer.bind(10);
        clusterStructuresBuffer.bind(11);
    }

    infoBuffer.bind(LIGHT_CLUSTER_INFO_BINDING_POINT);
    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);

    {
        auto tim = timer->CreateScope("GPU Light Assignment");

        lightAssignmentShader->bind();
        lightAssignmentShader->dispatchCompute(gridCount[0], gridCount[1], gridCount[2]);
        lightAssignmentShader->memoryBarrier(MemoryBarrierMask::GL_BUFFER_UPDATE_BARRIER_BIT);
        lightAssignmentShader->unbind();
    }

    assert_no_glerror();
}

void GPUAssignmentClusterer::buildClusters(Camera* cam)
{
    clustersDirty = false;
    float camNear = cam->zNear;
    float camFar  = cam->zFar;
    mat4 invProjection(inverse(cam->proj));

    clusterInfoBuffer.screenSpaceTileSize = screenSpaceTileSize;
    clusterInfoBuffer.screenWidth         = width;
    clusterInfoBuffer.screenHeight        = height;

    clusterInfoBuffer.zNear = cam->zNear;
    clusterInfoBuffer.zFar  = cam->zFar;

    gridCount[0] = std::ceil((float)width / (float)screenSpaceTileSize);
    gridCount[1] = std::ceil((float)height / (float)screenSpaceTileSize);
    if (clusterThreeDimensional)
        gridCount[2] = depthSplits + 1;
    else
        gridCount[2] = 1;

    clusterInfoBuffer.clusterX = (int)gridCount[0];
    clusterInfoBuffer.clusterY = (int)gridCount[1];

    clusterInfoBuffer.scale = gridCount[2] / log2(camFar / camNear);
    clusterInfoBuffer.bias  = -(gridCount[2] * log2(camNear) / log2(camFar / camNear));

    // Calculate Cluster Planes in View Space.
    int clusterCount = (int)(gridCount[0] * gridCount[1] * gridCount[2]);
    clusterBuffer.clusterList.clear();
    clusterBuffer.clusterList.resize(clusterCount);
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
                vec4 screenSpaceBL(x * screenSpaceTileSize, y * screenSpaceTileSize, -1.0, 1.0);  // Bottom left
                vec4 screenSpaceTR((x + 1) * screenSpaceTileSize, (y + 1) * screenSpaceTileSize, -1.0,
                                   1.0);  // Top Right

                // Doom Depth Split, because it looks good.
                float tileNear = -camNear * pow(camFar / camNear, (float)z / gridCount[2]);
                float tileFar  = -camNear * pow(camFar / camNear, (float)(z + 1) / gridCount[2]);

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
        auto tim                     = timer->CreateScope("Info Update");
        clusterInfoBuffer.tileDebug  = screenSpaceDebug ? allowedItemsPerCluster : 0;
        clusterInfoBuffer.splitDebug = splitDebug ? 1 : 0;

        itemBuffer.itemList.clear();
        itemBuffer.itemList.resize(allowedItemsPerCluster * clusterInfoBuffer.clusterListCount);
        clusterInfoBuffer.itemListCount = 0;  // itemBuffer.itemList.size();

        int itemBufferSize = sizeof(itemBuffer) + sizeof(clusterItem) * itemBuffer.itemList.size();
        int maxBlockSize   = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
        SAIGA_ASSERT(maxBlockSize > itemBufferSize, "Item SSB size too big!");

        itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), itemBufferSize, GL_DYNAMIC_DRAW);

        int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
        clusterListBuffer.createGLBuffer(clusterBuffer.clusterList.data(), clusterListSize, GL_DYNAMIC_DRAW);

        int clusterStructuresSize = sizeof(clusterBounds) * cullingCluster.size();
        clusterStructuresBuffer.createGLBuffer(cullingCluster.data(), clusterStructuresSize, GL_DYNAMIC_DRAW);


        infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);
    }
}

}  // namespace Saiga
