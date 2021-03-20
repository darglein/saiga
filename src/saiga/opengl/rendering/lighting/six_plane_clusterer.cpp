/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/six_plane_clusterer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

namespace Saiga
{
SixPlaneClusterer::SixPlaneClusterer(GLTimerSystem* timer, ClustererParameters _params) : Clusterer(timer, _params) {}

SixPlaneClusterer::~SixPlaneClusterer() {}

void SixPlaneClusterer::clusterLightsInternal(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    clustersDirty |= cam->proj != cached_projection;
    cached_projection = cam->proj;

    if (clustersDirty) buildClusters(cam);

    lightAssignmentTimer.start();

    // memset(itemBuffer.itemList.data(), 0, sizeof(clusterItem) * itemBuffer.itemList.size());
    const int maxClusterItemsPerCluster = 512;  // TODO Paul: Hardcoded?
    int16_t visibleLightIndices[maxClusterItemsPerCluster];

    int globalOffset = 0;

    for (int c = 0; c < culling_cluster.size(); ++c)
    {
        const auto& cluster_planes = culling_cluster[c].planes;

        int visiblePLCount = 0;
        for (int i = 0; i < pointLightsClusterData.size(); ++i)
        {
            PointLightClusterData& plc = pointLightsClusterData[i];
            bool intersection          = true;
            vec3 sphereCenter          = cam->WorldToView(plc.world_center);
            for (int p = 0; p < 6; ++p)
            {
                if (dot(cluster_planes[p].normal, sphereCenter) - cluster_planes[p].d + plc.radius < 0.0)
                {
                    intersection = false;
                    break;
                }
            }
            if (intersection)
            {
                if (visiblePLCount >= maxClusterItemsPerCluster) break;

                visibleLightIndices[visiblePLCount] = i;
                visiblePLCount++;
            }
        }

        int visibleSLCount = 0;
        for (int i = 0; i < spotLightsClusterData.size(); ++i)
        {
            SpotLightClusterData& plc = spotLightsClusterData[i];
            bool intersection         = true;
            vec3 sphereCenter         = cam->WorldToView(plc.world_center);
            for (int p = 0; p < 6; ++p)
            {
                if (dot(cluster_planes[p].normal, sphereCenter) - cluster_planes[p].d + plc.radius < 0.0)
                {
                    intersection = false;
                    break;
                }
            }
            if (intersection)
            {
                if (visiblePLCount + visibleSLCount >= maxClusterItemsPerCluster) break;

                visibleLightIndices[visiblePLCount + visibleSLCount] = i;
                visibleSLCount++;
            }
        }

        cluster& gpuCluster = clusterBuffer.clusterList.at(c);
        gpuCluster.offset   = globalOffset;
        globalOffset += std::ceil(visiblePLCount * 0.5f);
        globalOffset += std::ceil(visibleSLCount * 0.5f);

        memcpy(&(itemBuffer.itemList[gpuCluster.offset]), &visibleLightIndices[0], maxClusterItemsPerCluster * sizeof(int16_t));

        gpuCluster.plCount = visiblePLCount;
        gpuCluster.slCount = visibleSLCount;
    }

    lightAssignmentTimer.stop();
    cpuAssignmentTimes[timerIndex] = lightAssignmentTimer.getTimeMS();
    timerIndex = (timerIndex + 1) % 100;


    {
        auto tim            = timer->CreateScope("ClusterUpdate");
        int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
        clusterListBuffer.updateBuffer(clusterBuffer.clusterList.data(), clusterListSize, 0);

        int itemListSize = sizeof(clusterItem) * itemBuffer.itemList.size();
        // std::cout << "Used " << globalOffset * sizeof(clusterItem) << " item slots of " << itemListSize << std::endl;
        itemListBuffer.updateBuffer(itemBuffer.itemList.data(), itemListSize, 0);

        infoBuffer.bind(LIGHT_CLUSTER_INFO_BINDING_POINT);
        clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
        itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);
    }
    assert_no_glerror();
}

void SixPlaneClusterer::buildClusters(Camera* cam)
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


    float gridCount[3];
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

    culling_cluster.clear();
    culling_cluster.resize(clusterCount);
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
                vec4 screenSpaceBL(x * screenSpaceTileSize, y * screenSpaceTileSize, -1.0, 1.0);        // Bottom left
                vec4 screenSpaceTL(x * screenSpaceTileSize, (y + 1) * screenSpaceTileSize, -1.0, 1.0);  // Top left
                vec4 screenSpaceBR((x + 1) * screenSpaceTileSize, y * screenSpaceTileSize, -1.0, 1.0);  // Bottom Right
                vec4 screenSpaceTR((x + 1) * screenSpaceTileSize, (y + 1) * screenSpaceTileSize, -1.0,
                                   1.0);  // Top Right

                // Doom Depth Split, because it looks good.
                float tileNear = -camNear * pow(camFar / camNear, (float)z / gridCount[2]);
                float tileFar  = -camNear * pow(camFar / camNear, (float)(z + 1) / gridCount[2]);

                vec3 viewNearPlaneBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
                vec3 viewNearPlaneTL(make_vec3(viewPosFromScreenPos(screenSpaceTL, invProjection)));
                vec3 viewNearPlaneBR(make_vec3(viewPosFromScreenPos(screenSpaceBR, invProjection)));
                vec3 viewNearPlaneTR(make_vec3(viewPosFromScreenPos(screenSpaceTR, invProjection)));

                vec3 viewNearClusterBL(zeroZIntersection(viewNearPlaneBL, tileNear));
                vec3 viewNearClusterTL(zeroZIntersection(viewNearPlaneTL, tileNear));
                vec3 viewNearClusterBR(zeroZIntersection(viewNearPlaneBR, tileNear));
                vec3 viewNearClusterTR(zeroZIntersection(viewNearPlaneTR, tileNear));

                vec3 viewFarClusterBL(zeroZIntersection(viewNearPlaneBL, tileFar));
                vec3 viewFarClusterTL(zeroZIntersection(viewNearPlaneTL, tileFar));
                vec3 viewFarClusterBR(zeroZIntersection(viewNearPlaneBR, tileFar));
                vec3 viewFarClusterTR(zeroZIntersection(viewNearPlaneTR, tileFar));

                Plane nearPlane(viewNearClusterBL, vec3(0, 0, -1));
                Plane farPlane(viewFarClusterTR, vec3(0, 0, 1));

                vec3 p0, p1, p2;

                p0 = viewNearClusterBL;
                p1 = viewFarClusterBL;
                p2 = viewFarClusterTL;
                Plane leftPlane(p0, p1, p2);

                p0 = viewNearClusterTL;
                p1 = viewFarClusterTL;
                p2 = viewFarClusterTR;
                Plane topPlane(p0, p1, p2);

                p0 = viewNearClusterTR;
                p1 = viewFarClusterTR;
                p2 = viewFarClusterBR;
                Plane rightPlane(p0, p1, p2);

                p0 = viewNearClusterBR;
                p1 = viewFarClusterBR;
                p2 = viewFarClusterBL;
                Plane bottomPlane(p0, p1, p2);


                int tileIndex = x + (int)gridCount[0] * y + (int)(gridCount[0] * gridCount[1]) * z;

                auto& planes = culling_cluster.at(tileIndex).planes;
                planes[0]    = nearPlane;
                planes[1]    = farPlane;
                planes[2]    = leftPlane;
                planes[3]    = rightPlane;
                planes[4]    = topPlane;
                planes[5]    = bottomPlane;


                if (clusterDebug)
                {
                    PointVertex v;

                    v.color = vec3(1.0, 0.75, 0);

                    v.position = viewNearClusterBL;
                    debugCluster.lines.push_back(v);
                    v.position = viewNearClusterTL;
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = viewNearClusterTR;
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = viewNearClusterBR;
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = viewNearClusterBL;
                    debugCluster.lines.push_back(v);

                    v.position = viewFarClusterBL;
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterTL;
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterTR;
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterBR;
                    debugCluster.lines.push_back(v);
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterBL;
                    debugCluster.lines.push_back(v);

                    v.position = viewNearClusterBL;
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterBL;
                    debugCluster.lines.push_back(v);

                    v.position = viewNearClusterTL;
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterTL;
                    debugCluster.lines.push_back(v);

                    v.position = viewNearClusterBR;
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterBR;
                    debugCluster.lines.push_back(v);

                    v.position = viewNearClusterTR;
                    debugCluster.lines.push_back(v);
                    v.position = viewFarClusterTR;
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
        auto tim = timer->CreateScope("Info Update");
        itemBuffer.itemList.clear();
        int maxClusterItemsPerCluster = 512;  // TODO Paul: Hardcoded?

        clusterInfoBuffer.tileDebug = screenSpaceDebug ? maxClusterItemsPerCluster : 0;
        itemBuffer.itemList.resize(maxClusterItemsPerCluster / 2 * clusterCount);
        clusterInfoBuffer.itemListCount = itemBuffer.itemList.size();

        int itemBufferSize = sizeof(itemBuffer) + sizeof(clusterItem) * itemBuffer.itemList.size();
        int maxBlockSize   = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
        SAIGA_ASSERT(maxBlockSize > itemBufferSize, "Item SSB size too big!");

        itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), itemBufferSize, GL_DYNAMIC_DRAW);

        int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
        clusterListBuffer.createGLBuffer(clusterBuffer.clusterList.data(), clusterListSize, GL_DYNAMIC_DRAW);

        infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);
    }
}

}  // namespace Saiga
