/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/cpu_plane_clusterer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/imgui/imgui_opengl.h"

namespace Saiga
{
CPUPlaneClusterer::CPUPlaneClusterer(GLTimerSystem* timer, const ClustererParameters& _params) : Clusterer(timer, _params)
{
    itemList.resize(1);
}

CPUPlaneClusterer::~CPUPlaneClusterer() {}

void CPUPlaneClusterer::clusterLightsInternal(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    clustersDirty |= cam->proj != cached_projection;
    cached_projection = cam->proj;

    if (clustersDirty) buildClusters(cam);

    int itemCount = 0;

    {
        lightAssignmentTimer.start();

        for (int c = 0; c < clusterList.size(); ++c)
        {
            clusterCache[c].clear();
            clusterCache[c].push_back(0);  // PL Count
        }


        if (lightsDebug && updateLightsDebug) lightClustersDebug.lines.clear();
        if (!params.SAT)
        {
            for (int i = 0; i < pointLightCount; ++i)
            {
                LightBoundingSphere& plc = lightsClusterData[i];
                vec3 sphereCenter          = cam->WorldToView(plc.world_center);
                clusterLoop(sphereCenter, plc.radius, i, true, itemCount);
            }
            for (int i = pointLightCount; i < lightsClusterData.size(); ++i)
            {
                LightBoundingSphere& slc = lightsClusterData[i];
                vec3 sphereCenter         = cam->WorldToView(slc.world_center);
                clusterLoop(sphereCenter, slc.radius, i, false, itemCount);
            }
        }
        else
        {
            for (int i = 0; i < pointLightCount; ++i)
            {
                auto& cData        = lightsClusterData[i];
                vec3 sphereCenter  = cam->WorldToView(cData.world_center);
                float sphereRadius = cData.radius;
                Sphere sphere(sphereCenter, sphereRadius);

                for (int x = 0; x < planesX.size() - 1; ++x)
                {
                    for (int y = 0; y < planesY.size() - 1; ++y)
                    {
                        for (int z = 0; z < planesZ.size() - 1; ++z)
                        {
                            int tileIndex = getTileIndex(x, y, z);

                            const Frustum& fr = debugFrusta[tileIndex];

                            if (fr.intersectSAT(sphere))
                            {
                                clusterCache[tileIndex].push_back(i);
                                clusterCache[tileIndex][0]++;
                                itemCount++;
                            }
                        }
                    }
                }
            }
            for (int i = pointLightCount; i < lightsClusterData.size(); ++i)
            {
                auto& cData        = lightsClusterData[i];
                vec3 sphereCenter  = cam->WorldToView(cData.world_center);
                float sphereRadius = cData.radius;
                Sphere sphere(sphereCenter, sphereRadius);

                for (int x = 0; x < planesX.size() - 1; ++x)
                {
                    for (int y = 0; y < planesY.size() - 1; ++y)
                    {
                        for (int z = 0; z < planesZ.size() - 1; ++z)
                        {
                            int tileIndex = getTileIndex(x, y, z);

                            const Frustum& fr = debugFrusta[tileIndex];

                            if (fr.intersectSAT(sphere))
                            {
                                clusterCache[tileIndex].push_back(i);
                                itemCount++;
                            }
                        }
                    }
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
            clusterInfoBuffer.tileDebug     = screenSpaceDebug ? itemList.size() : 0;
            clusterInfoBuffer.splitDebug    = splitDebug ? 1 : 0;

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

            if (lightsDebug && updateLightsDebug && (gpuCluster.plCount > 0 || gpuCluster.slCount > 0))
            {
                const auto& dbg = debugFrusta[c];
                PointVertex v;
                v.color = vec3(1, 1, 1);

                v.position = dbg.vertices[2];
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[0];
                lightClustersDebug.lines.push_back(v);
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[1];
                lightClustersDebug.lines.push_back(v);
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[3];
                lightClustersDebug.lines.push_back(v);
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[2];
                lightClustersDebug.lines.push_back(v);

                v.position = dbg.vertices[6];
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[4];
                lightClustersDebug.lines.push_back(v);
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[5];
                lightClustersDebug.lines.push_back(v);
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[7];
                lightClustersDebug.lines.push_back(v);
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[6];
                lightClustersDebug.lines.push_back(v);

                v.position = dbg.vertices[2];
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[6];
                lightClustersDebug.lines.push_back(v);

                v.position = dbg.vertices[0];
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[4];
                lightClustersDebug.lines.push_back(v);

                v.position = dbg.vertices[3];
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[7];
                lightClustersDebug.lines.push_back(v);

                v.position = dbg.vertices[1];
                lightClustersDebug.lines.push_back(v);
                v.position = dbg.vertices[5];
                lightClustersDebug.lines.push_back(v);
            }
        }

        if (lightsDebug && updateLightsDebug)
        {
            lightClustersDebug.lineWidth = 3;

            lightClustersDebug.setModelMatrix(cam->getModelMatrix());  // is inverse view.
            lightClustersDebug.translateLocal(vec3(0, 0, -0.0001f));
#if 0
        lightClustersDebug.setPosition(make_vec4(0));
        lightClustersDebug.translateGlobal(vec3(0, 6, 0));
        lightClustersDebug.setScale(make_vec3(0.33f));
#endif
            lightClustersDebug.calculateModel();
            lightClustersDebug.updateBuffer();
            updateLightsDebug = false;
        }

        lightAssignmentTimer.stop();
        cpuAssignmentTimes[timerIndex] = lightAssignmentTimer.getTimeMS();
        timerIndex                     = (timerIndex + 1) % 100;
    }

    {
        auto tim            = timer->Measure("clusterupdate");
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

void CPUPlaneClusterer::clusterLoop(vec3 sphereCenter, float sphereRadius, int index, bool pl, int& itemCount)
{
    int maxDepthCluster = planesZ.size() - 2;

    int x0 = 0, x1 = planesX.size() - 1;
    int y0 = 0, y1 = planesY.size() - 1;
    int z0 = 0, z1 = planesZ.size() - 1;

    int centerOutsideZ = 0;
    int centerOutsideY = 0;

    while (z0 <= z1 && planesZ[z0].distance(sphereCenter) >= sphereRadius)
    {
        z0++;
    }
    if (--z0 < 0 && planesZ[0].distance(sphereCenter) < 0)
    {
        centerOutsideZ--;  // Center is behind camera far plane.
    }
    z0 = std::max(0, z0);
    while (z1 >= z0 && -planesZ[z1].distance(sphereCenter) >= sphereRadius)
    {
        --z1;
    }
    if (++z1 > (int)planesZ.size() - 1 && planesZ[(int)planesZ.size() - 1].distance(sphereCenter) > 0)
    {
        centerOutsideZ++;  // Center is in front of camera near plane.
    }
    z1 = std::min(z1, (int)planesZ.size() - 1);
    if (z0 >= z1)
    {
        return;
    }


    while (y0 <= y1 && planesY[y0].distance(sphereCenter) >= sphereRadius)
    {
        y0++;
    }
    if (--y0 < 0 && planesY[0].distance(sphereCenter) < 0)
    {
        centerOutsideY--;  // Center left outside frustum.
    }
    y0 = std::max(0, y0);
    while (y1 >= y0 && -planesY[y1].distance(sphereCenter) >= sphereRadius)
    {
        --y1;
    }
    if (++y1 > (int)planesY.size() - 1 && planesY[(int)planesY.size() - 1].distance(sphereCenter) > 0)
    {
        centerOutsideY++;  // Center right outside frustum.
    }
    y1 = std::min(y1, (int)planesY.size() - 1);
    if (y0 >= y1)
    {
        return;
    }


    while (x0 <= x1 && planesX[x0].distance(sphereCenter) >= sphereRadius)
    {
        x0++;
    }
    --x0;
    x0 = std::max(0, x0);
    while (x1 >= x0 && -planesX[x1].distance(sphereCenter) >= sphereRadius)
    {
        --x1;
    }
    ++x1;
    x1 = std::min(x1, (int)planesX.size() - 1);
    if (x0 >= x1)
    {
        return;
    }



    if (!params.refinement)
    {
        // This is without the sphere refinement
        for (int z = z0; z < z1; ++z)
        {
            for (int y = y0; y < y1; ++y)
            {
                for (int x = x0; x < x1; ++x)
                {
                    int tileIndex = getTileIndex(x, y, maxDepthCluster - z);

                    clusterCache[tileIndex].push_back(index);
                    itemCount++;
                    if (pl) clusterCache[tileIndex][0]++;
                }
            }
        }
    }
    else
    {
        if (centerOutsideZ < 0)
        {
            z0 = -(int)planesZ.size() * 4;
        }
        if (centerOutsideZ > 0)
        {
            z1 = (int)planesZ.size() * 4;
        }
        int cz      = (z0 + z1);
        int centerZ = cz / 2;
        if (centerOutsideZ == 0 && cz % 2 == 0)
        {
            float d0 = planesZ[centerZ].distance(sphereCenter);
            if (d0 < 1e-5f) centerZ -= 1;
        }

        if (centerOutsideY < 0)
        {
            y0 = -(int)planesY.size() * 4;
        }
        if (centerOutsideY > 0)
        {
            y1 = (int)planesY.size() * 4;
        }
        int cy      = (y0 + y1);
        int centerY = cy / 2;
        if (centerOutsideY == 0 && cy % 2 == 0)
        {
            float d0 = planesY[centerY].distance(sphereCenter);
            if (d0 < 1e-5f) centerY -= 1;
        }

        Sphere lightSphere(sphereCenter, sphereRadius);

        z0 = std::max(0, z0);
        z1 = std::min(z1, (int)planesZ.size() - 1);
        y0 = std::max(0, y0);
        y1 = std::min(y1, (int)planesY.size() - 1);

        for (int z = z0; z < z1; ++z)
        {
            Sphere zLight = lightSphere;
            if (z != centerZ)
            {
                Plane plane = (z < centerZ) ? planesZ[z + 1] : planesZ[z].invert();
                auto circle = plane.intersectingCircle(zLight.pos, zLight.r);
                zLight.pos  = circle.first;
                zLight.r    = circle.second;
                if (zLight.r < 1e-5) continue;
            }
            for (int y = y0; y < y1; ++y)
            {
                Sphere yLight = zLight;
                if (y != centerY)
                {
                    Plane plane = (y < centerY) ? planesY[y + 1] : planesY[y].invert();
                    auto circle = plane.intersectingCircle(yLight.pos, yLight.r);
                    yLight.pos  = circle.first;
                    yLight.r    = circle.second;
                    if (yLight.r < 1e-5) continue;
                }


                int x = x0;
                while (x < x1 && planesX[x].distance(yLight.pos) >= yLight.r) x++;
                --x;
                x      = std::max(x0, x);
                int xs = x1;
                while (xs >= x && -planesX[xs].distance(yLight.pos) >= yLight.r) --xs;
                ++xs;
                xs = std::min(xs, x1);

                for (; x < xs; ++x)
                {
                    int tileIndex = getTileIndex(x, y, maxDepthCluster - z);

                    clusterCache[tileIndex].push_back(index);
                    itemCount++;
                    if (pl) clusterCache[tileIndex][0]++;
                }
            }
        }
    }
}

void CPUPlaneClusterer::buildClusters(Camera* cam)
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

    planesX.clear();
    planesY.clear();
    planesZ.clear();
    planesX.resize((int)gridCount[0] + 1);
    planesY.resize((int)gridCount[1] + 1);
    planesZ.resize((int)gridCount[2] + 1);
    if (clusterDebug && updateDebug)
    {
        debugCluster.lines.clear();
    }

    // const vec3 eyeView(0.0); Not required because it is zero.
    PointVertex v;

    v.color = vec3(1.0, 0.75, 0);

    vec3 p0, p1, p2;

    for (int x = 0; x < planesX.size(); ++x)
    {
        vec4 screenSpaceBL(x * params.screenSpaceTileSize, 0, -1.0, 1.0);       // Bottom left
        vec4 screenSpaceTL(x * params.screenSpaceTileSize, height, -1.0, 1.0);  // Top left

        vec3 viewBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
        vec3 viewTL(make_vec3(viewPosFromScreenPos(screenSpaceTL, invProjection)));

        vec3 viewNearPlaneBL(zeroZIntersection(viewBL, -camNear));
        // vec3 viewNearPlaneTL(zeroZIntersection(viewTL, -camNear));

        vec3 viewFarPlaneBL(zeroZIntersection(viewBL, -camFar));
        vec3 viewFarPlaneTL(zeroZIntersection(viewTL, -camFar));

        p0 = viewNearPlaneBL;
        p1 = viewFarPlaneBL;
        p2 = viewFarPlaneTL;

        planesX[x] = Plane(p0, p1, p2);
    }

    for (int y = 0; y < planesY.size(); ++y)
    {
        vec4 screenSpaceBL(0, y * params.screenSpaceTileSize, -1.0, 1.0);      // Bottom left
        vec4 screenSpaceBR(width, y * params.screenSpaceTileSize, -1.0, 1.0);  // Bottom Right

        vec3 viewBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
        vec3 viewBR(make_vec3(viewPosFromScreenPos(screenSpaceBR, invProjection)));

        // vec3 viewNearPlaneBL(zeroZIntersection(viewBL, -camNear));
        vec3 viewNearPlaneBR(zeroZIntersection(viewBR, -camNear));

        vec3 viewFarPlaneBL(zeroZIntersection(viewBL, -camFar));
        vec3 viewFarPlaneBR(zeroZIntersection(viewBR, -camFar));

        p0 = viewNearPlaneBR;
        p1 = viewFarPlaneBR;
        p2 = viewFarPlaneBL;

        planesY[y] = Plane(p0, p1, p2);
    }

    // planesZ.size is gridCount[2] + 1
    for (int z = 0; z < planesZ.size(); ++z)
    {
        vec4 screenSpaceBL(0, 0, -1.0, 1.0);

        vec3 viewNearPlaneBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));

        // Doom Depth Split, because it looks good.
        // float tileNear = -camNear * pow(camFar / camNear, (float)z / gridCount[2]);
        float tileFar;
        if (useSpecialNear && z == planesZ.size() - 1)
        {
            tileFar = -camNear;
        }
        else
        {
            int calcZ = useSpecialNear ? (gridCount[2] - (float)z) - 1 : (gridCount[2] - (float)z);

            // Doom Depth Split, because it looks good.
            tileFar = -(camNear + specialNearDepth) *
                      pow(camFar / (camNear + specialNearDepth), (float)calcZ / specialGridCount);
        }

        vec3 viewFarClusterBL(zeroZIntersection(viewNearPlaneBL, tileFar));

        planesZ[z] = Plane(viewFarClusterBL, vec3(0, 0, 1));
    }

    if (params.SAT || clusterDebug || lightsDebug)
    {
        debugFrusta.resize(clusterCount);
        for (int x = 0; x < (int)gridCount[0]; ++x)
        {
            for (int y = 0; y < (int)gridCount[1]; ++y)
            {
                for (int z = 0; z < (int)gridCount[2]; ++z)
                {
                    vec4 screenSpaceBL(x * params.screenSpaceTileSize, y * params.screenSpaceTileSize, -1.0, 1.0);  // Bottom left
                    vec4 screenSpaceTL(x * params.screenSpaceTileSize, (y + 1) * params.screenSpaceTileSize, -1.0, 1.0);  // Top left
                    vec4 screenSpaceBR((x + 1) * params.screenSpaceTileSize, y * params.screenSpaceTileSize, -1.0,
                                       1.0);  // Bottom Right
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

                    int tileIndex = getTileIndex(x, y, z);

                    /*
                    Plane nearPlane(viewNearClusterBL, vec3(0, 0, 1));
                    Plane farPlane(viewFarClusterTR, vec3(0, 0, -1));

                    vec3 p0, p1, p2;

                    p0 = viewNearClusterBL;
                    p1 = viewFarClusterBL;
                    p2 = viewFarClusterTL;
                    Plane leftPlane(p0, p2, p1);

                    p0 = viewNearClusterTL;
                    p1 = viewFarClusterTL;
                    p2 = viewFarClusterTR;
                    Plane topPlane(p0, p2, p1);

                    p0 = viewNearClusterTR;
                    p1 = viewFarClusterTR;
                    p2 = viewFarClusterBR;
                    Plane rightPlane(p0, p2, p1);

                    p0 = viewNearClusterBR;
                    p1 = viewFarClusterBR;
                    p2 = viewFarClusterBL;
                    Plane bottomPlane(p0, p2, p1);
                    */

                    Frustum& fr = debugFrusta.at(tileIndex);

                    // fr.planes[0] = nearPlane;
                    // fr.planes[1] = farPlane;
                    // fr.planes[2] = topPlane;
                    // fr.planes[3] = bottomPlane;
                    // fr.planes[5] = leftPlane;
                    // fr.planes[4] = rightPlane;

                    fr.planes[0] = planesZ[((int)gridCount[2] - z - 1)].invert();
                    fr.planes[1] = planesZ[((int)gridCount[2] - z)];
                    fr.planes[2] = planesY[y + 1];
                    fr.planes[3] = planesY[y].invert();
                    fr.planes[4] = planesX[x].invert();
                    fr.planes[5] = planesX[x + 1];

                    fr.vertices[0] = viewNearClusterTL;
                    fr.vertices[1] = viewNearClusterTR;
                    fr.vertices[2] = viewNearClusterBL;
                    fr.vertices[3] = viewNearClusterBR;
                    fr.vertices[4] = viewFarClusterTL;
                    fr.vertices[5] = viewFarClusterTR;
                    fr.vertices[6] = viewFarClusterBL;
                    fr.vertices[7] = viewFarClusterBR;

                    if (clusterDebug && updateDebug)
                    {
                        PointVertex v;

                        v.color = vec3(0.5, 0.125, 0);

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
    }

    if (clusterDebug && updateDebug)
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

        infoBuffer.update(clusterInfoBuffer);
    }
}

void CPUPlaneClusterer::imgui()
{
    Clusterer::imgui();
    if (ImGui::Begin("Clusterer"))
    {
        clustersDirty |= ImGui::Checkbox("refinement", &params.refinement);

        ImGui::Text("ItemListSize: %d KB", int(itemList.size() * sizeof(ClusterItem) * 0.001f));

        if (ImGui::Checkbox("lightsDebug", &lightsDebug) && lightsDebug)
        {
            updateLightsDebug = true;
            clustersDirty     = true;
        }
        if (lightsDebug)
            if (ImGui::Button("updateLightsDebug")) updateLightsDebug = true;

        clustersDirty |= ImGui::Checkbox("SAT Debug", &params.SAT);
    }
    ImGui::End();
}

}  // namespace Saiga
