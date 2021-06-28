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
SixPlaneClusterer::SixPlaneClusterer(GLTimerSystem* timer, const ClustererParameters& _params) : Clusterer(timer, _params)
{
    itemList.resize(1);
}

SixPlaneClusterer::~SixPlaneClusterer() {}

void SixPlaneClusterer::clusterLightsInternal(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    clustersDirty |= cam->proj != cached_projection;
    cached_projection = cam->proj;

    if (clustersDirty) buildClusters(cam);

    lightAssignmentTimer.start();

    for (int c = 0; c < clusterList.size(); ++c)
    {
        clusterCache[c].clear();
        clusterCache[c].push_back(0);  // PL Count
    }

    int itemCount = 0;

    for (int i = 0; i < pointLightCount; ++i)
    {
        LightBoundingSphere& plc = lightsClusterData[i];
        vec3 sphereCenter          = cam->WorldToView(plc.world_center);
        for (int c = 0; c < cullingCluster.size(); ++c)
        {
            bool intersection          = true;
            const auto& cluster_planes = cullingCluster[c].planes;
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
                clusterCache[c].push_back(i);
                clusterCache[c][0]++;
                itemCount++;
            }
        }
    }

    for (int i = pointLightCount; i < lightsClusterData.size(); ++i)
    {
        LightBoundingSphere& plc = lightsClusterData[i];
        vec3 sphereCenter         = cam->WorldToView(plc.world_center);
        for (int c = 0; c < cullingCluster.size(); ++c)
        {
            bool intersection          = true;
            const auto& cluster_planes = cullingCluster[c].planes;
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
                clusterCache[c].push_back(i);
                itemCount++;
            }
        }
    }

    bool adaptSize = false;
    if (itemCount > itemList.size())
        {
            adaptSize = true;
            do
            {
                itemList.resize(itemList.size() * 2);
            } while (itemCount > itemList.size());
        }
        if (itemCount < itemList.size() * 0.5 && itemList.size() > 2)
        {
            adaptSize = true;
            do
            {
                itemList.resize(itemList.size() / 2);
            } while (itemCount < itemList.size() * 0.5 && itemList.size() > 2);
        }

        if (adaptSize)
        {
            auto tim = timer->Measure("Info Update");

            clusterInfoBuffer.itemListCount = itemList.size();
            clusterInfoBuffer.tileDebug  = screenSpaceDebug ? itemList.size() : 0;
            clusterInfoBuffer.splitDebug = splitDebug ? 1 : 0;

            int itemListSize = sizeof(ClusterItem) * itemList.size();
            int maxBlockSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
            SAIGA_ASSERT(maxBlockSize > itemListSize, "Item SSB size too big!");

            infoBuffer.update(clusterInfoBuffer);
        }

    int globalOffset = 0;

    for (int c = 0; c < clusterCache.size(); ++c)
    {
        auto cl             = clusterCache[c];
        Cluster& gpuCluster = clusterList.at(c);

        gpuCluster.offset = globalOffset;
        SAIGA_ASSERT(gpuCluster.offset < itemList.size(), "Too many items!");
        gpuCluster.plCount = cl[0];
        gpuCluster.slCount = cl.size() - 1 - cl[0];
        globalOffset += gpuCluster.plCount;
        globalOffset += gpuCluster.slCount;
        if (cl.size() < 2)
        {
            continue;
        }

        memcpy(&(itemList[gpuCluster.offset]), &cl[1], (cl.size() - 1) * sizeof(ClusterItem));
    }

    lightAssignmentTimer.stop();
    cpuAssignmentTimes[timerIndex] = lightAssignmentTimer.getTimeMS();
    timerIndex                     = (timerIndex + 1) % 100;

    {
        auto tim = timer->Measure("ClusterUpdate");
        if (clusterList.size() > clusterListBuffer.Size())
        {
            clusterListBuffer.create(clusterList, GL_DYNAMIC_DRAW);
            clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
        }
        else
        {
            clusterListBuffer.update(clusterList);
        }

        if (itemList.size() > itemListBuffer.Size())
        {
            itemListBuffer.create(itemList, GL_DYNAMIC_DRAW);
            itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);
        }
        else
        {
            itemListBuffer.update(itemList);
        }
    }

    assert_no_glerror();
}

void SixPlaneClusterer::buildClusters(Camera* cam)
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


    float gridCount[3];
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
    clusterList.clear();
    clusterList.resize(clusterCount);
    clusterInfoBuffer.clusterListCount = clusterCount;
    clusterCache.clear();
    clusterCache.resize(clusterCount);

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
                vec4 screenSpaceBL(x * params.screenSpaceTileSize, y * params.screenSpaceTileSize, -1.0, 1.0);        // Bottom left
                vec4 screenSpaceTL(x * params.screenSpaceTileSize, (y + 1) * params.screenSpaceTileSize, -1.0, 1.0);  // Top left
                vec4 screenSpaceBR((x + 1) * params.screenSpaceTileSize, y * params.screenSpaceTileSize, -1.0, 1.0);  // Bottom Right
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

                auto& planes = cullingCluster.at(tileIndex).planes;
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
        auto tim                     = timer->Measure("Info Update");
        clusterInfoBuffer.tileDebug  = screenSpaceDebug ? itemList.size() : 0;
        clusterInfoBuffer.splitDebug = splitDebug ? 1 : 0;

        itemList.clear();
        itemList.resize(1);
        clusterInfoBuffer.itemListCount = itemList.size();

        int itemListSize = sizeof(ClusterItem) * itemList.size();
        int maxBlockSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
        SAIGA_ASSERT(maxBlockSize > itemListSize, "Item SSB size too big!");

        infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);
    }
}

}  // namespace Saiga
