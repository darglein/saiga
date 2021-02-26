/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/cpu_plane_clusterer.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{

CPUPlaneClusterer::CPUPlaneClusterer(ClustererParameters _params) : Clusterer(_params) {}

CPUPlaneClusterer::~CPUPlaneClusterer() {}

void CPUPlaneClusterer::clusterLightsInternal(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    clustersDirty |= cam->proj != cached_projection;
    cached_projection = cam->proj;

    if (clustersDirty) buildClusters(cam);

    lightAssignmentTimer.start();

    for (int c = 0; c < clusterBuffer.clusterList.size(); ++c)
    {
        clusterCache[c].clear();
    }

    int itemCount     = 0;
    int clusterCountZ = planesZ.size() - 2;

    if (renderDebugEnabled && debugFrustumToView) debugLights.lines.clear();

    for (int i = 0; i < pointLightsClusterData.size(); ++i)
    {
        PointLightClusterData& plc = pointLightsClusterData[i];
        vec3 sphereCenter          = cam->projectToViewSpace(plc.world_center);
        float sphereRadius         = plc.radius;

        int x0 = 0, x1 = planesX.size() - 1;
        int y0 = 0, y1 = planesY.size() - 1;
        int z0 = 0, z1 = planesZ.size() - 1;

        int centerOutsideZ = 0;
        int centerOutsideY = 0;


        while (z0 < z1 && planesZ[z0].distance(sphereCenter) >= sphereRadius)
        {
            z0++;
        }
        if (--z0 < 0 && planesZ[0].distance(sphereCenter) < 0)
        {
            centerOutsideZ--;
        }
        z0 = std::max(0, z0);
        while (z1 >= z0 && -planesZ[z1].distance(sphereCenter) >= sphereRadius)
        {
            --z1;
        }
        if (++z1 > (int)planesZ.size() - 1 && planesZ[(int)planesZ.size() - 1].distance(sphereCenter) > 0)
        {
            centerOutsideZ++;
        }
        z1 = std::min(z1, (int)planesZ.size() - 1);
        if (z0 >= z1)
        {
            continue;
        }


        while (y0 < y1 && planesY[y0].distance(sphereCenter) >= sphereRadius)
        {
            y0++;
        }
        if (--y0 < 0 && planesY[0].distance(sphereCenter) < 0)
        {
            centerOutsideY--;
        }
        y0 = std::max(0, y0);
        while (y1 >= y0 && -planesY[y1].distance(sphereCenter) >= sphereRadius)
        {
            --y1;
        }
        if (++y1 > (int)planesY.size() - 1 && planesY[(int)planesY.size() - 1].distance(sphereCenter) > 0)
        {
            centerOutsideY++;
        }
        y1 = std::min(y1, (int)planesY.size() - 1);
        if (y0 >= y1)
        {
            continue;
        }


        while (x0 < x1 && planesX[x0].distance(sphereCenter) >= sphereRadius)
        {
            x0++;
        }
        x0 = std::max(0, --x0);
        while (x1 >= x0 && -planesX[x1].distance(sphereCenter) >= sphereRadius)
        {
            --x1;
        }
        x1 = std::min(++x1, (int)planesX.size() - 1);
        if (x0 >= x1)
        {
            continue;
        }


        if (!refinement)
        {
            // This is without the sphere refinement
            for (int z = z0; z < z1; ++z)
            {
                for (int y = y0; y < y1; ++y)
                {
                    for (int x = x0; x < x1; ++x)
                    {
                        int tileIndex = x + clusterInfoBuffer.clusterX * y +
                                        (clusterInfoBuffer.clusterX * clusterInfoBuffer.clusterY) * (clusterCountZ - z);

                        clusterCache[tileIndex].push_back(i);
                        itemCount++;
                    }
                }
            }
        }
        else
        {
            if (centerOutsideZ < 0)
            {
                z0 = -(int)planesZ.size() * 2;
            }
            if (centerOutsideZ > 0)
            {
                z1 = (int)planesZ.size() * 2;
            }
            int cz      = (z0 + z1);
            int centerZ = cz / 2;
            if (centerOutsideZ == 0 && cz % 2 == 0)
            {
                float d0 = planesZ[z0].distance(sphereCenter);
                float d1 = -planesZ[z1].distance(sphereCenter);
                if (d0 <= d1) centerZ -= 1;
            }

            if (centerOutsideY < 0)
            {
                y0 = -(int)planesY.size() * 2;
            }
            if (centerOutsideY > 0)
            {
                y1 = (int)planesY.size() * 2;
            }
            int cy      = (y0 + y1);
            int centerY = cy / 2;
            if (centerOutsideY == 0 && cy % 2 == 0)
            {
                float d0 = planesY[y0].distance(sphereCenter);
                float d1 = -planesY[y1].distance(sphereCenter);
                if (d0 <= d1) centerY -= 1;
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
                    vec4 circle = plane.intersectingCircle(zLight.pos, zLight.r);
                    zLight.pos  = make_vec3(circle);
                    zLight.r    = circle.w();
                    if (zLight.r < 1e-5) continue;
                }
                for (int y = y0; y < y1; ++y)
                {
                    Sphere yLight = zLight;
                    if (y != centerY)
                    {
                        Plane plane = (y < centerY) ? planesY[y + 1] : planesY[y].invert();
                        vec4 circle = plane.intersectingCircle(yLight.pos, yLight.r);
                        yLight.pos  = make_vec3(circle);
                        yLight.r    = circle.w();
                        yLight.r    = circle.w();
                        if (yLight.r < 1e-5) continue;
                    }
                    int x = x0;
                    while (x < x1 && planesX[x].distance(yLight.pos) >= yLight.r) x++;
                    x      = std::max(x0, --x);
                    int xs = x1;
                    while (xs >= x && -planesX[xs].distance(yLight.pos) >= yLight.r) --xs;
                    xs = std::min(++xs, x1);

                    for (x; x < xs; ++x)
                    {
                        int tileIndex = x + clusterInfoBuffer.clusterX * y +
                                        (clusterInfoBuffer.clusterX * clusterInfoBuffer.clusterY) * (clusterCountZ - z);

                        clusterCache[tileIndex].push_back(i);
                        itemCount++;
                    }
                }
            }
        }
    }

    if (itemCount > itemBuffer.itemList.size())
    {
        do
        {
            avgAllowedItemsPerCluster *= 2;
            itemBuffer.itemList.resize(avgAllowedItemsPerCluster * clusterInfoBuffer.clusterListCount);
        } while (itemCount > itemBuffer.itemList.size());

        clusterInfoBuffer.itemListCount = itemBuffer.itemList.size();
        clusterInfoBuffer.tileDebug     = tileDebugView ? avgAllowedItemsPerCluster : 0;

        int itemBufferSize = sizeof(itemBuffer) + sizeof(clusterItem) * itemBuffer.itemList.size();
        int maxBlockSize   = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
        SAIGA_ASSERT(maxBlockSize > itemBufferSize, "Item SSB size too big!");

        itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), itemBufferSize);

        infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);
    }

    int globalOffset = 0;

    for (int c = 0; c < clusterCache.size(); ++c)
    {
        auto cl             = clusterCache[c];
        cluster& gpuCluster = clusterBuffer.clusterList.at(c);

        gpuCluster.offset  = globalOffset;
        gpuCluster.plCount = cl.size();
        gpuCluster.slCount = 0;
        SAIGA_ASSERT(gpuCluster.offset + gpuCluster.plCount < itemBuffer.itemList.size(), "Too many items!");
        globalOffset += gpuCluster.plCount;

        memcpy(&(itemBuffer.itemList[gpuCluster.offset]), cl.data(), cl.size() * sizeof(clusterItem));

        if (renderDebugEnabled && debugFrustumToView && gpuCluster.plCount > 0)
        {
            auto& dbg = debugPoints[c];
            PointVertex v;
            v.color = vec3(1, 0, 0);

            v.position = dbg.nBL;
            debugLights.lines.push_back(v);
            v.position = dbg.nTL;
            debugLights.lines.push_back(v);
            debugLights.lines.push_back(v);
            v.position = dbg.nTR;
            debugLights.lines.push_back(v);
            debugLights.lines.push_back(v);
            v.position = dbg.nBR;
            debugLights.lines.push_back(v);
            debugLights.lines.push_back(v);
            v.position = dbg.nBL;
            debugLights.lines.push_back(v);

            v.position = dbg.fBL;
            debugLights.lines.push_back(v);
            v.position = dbg.fTL;
            debugLights.lines.push_back(v);
            debugLights.lines.push_back(v);
            v.position = dbg.fTR;
            debugLights.lines.push_back(v);
            debugLights.lines.push_back(v);
            v.position = dbg.fBR;
            debugLights.lines.push_back(v);
            debugLights.lines.push_back(v);
            v.position = dbg.fBL;
            debugLights.lines.push_back(v);

            v.position = dbg.nBL;
            debugLights.lines.push_back(v);
            v.position = dbg.fBL;
            debugLights.lines.push_back(v);

            v.position = dbg.nTL;
            debugLights.lines.push_back(v);
            v.position = dbg.fTL;
            debugLights.lines.push_back(v);

            v.position = dbg.nBR;
            debugLights.lines.push_back(v);
            v.position = dbg.fBR;
            debugLights.lines.push_back(v);

            v.position = dbg.nTR;
            debugLights.lines.push_back(v);
            v.position = dbg.fTR;
            debugLights.lines.push_back(v);
        }
    }


    if (renderDebugEnabled && debugFrustumToView)
    {
        debugLights.lineWidth = 1;

        debugLights.setModelMatrix(cam->getModelMatrix());  // is inverse view.
        debugLights.translateLocal(vec3(0, 0, -0.0001f));
#if 0
        debugLights.setPosition(make_vec4(0));
        debugLights.translateGlobal(vec3(0, 6, 0));
        debugLights.setScale(make_vec3(0.33f));
#endif
        debugLights.calculateModel();
        debugLights.updateBuffer();
        debugFrustumToView = false;
    }

    lightAssignmentTimer.stop();

    startTimer(1);
    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.updateBuffer(clusterBuffer.clusterList.data(), clusterListSize, 0);

    int itemListSize = sizeof(clusterItem) * itemCount;
    // std::cout << "Used " << globalOffset * sizeof(clusterItem) << " item slots of " << itemListSize << std::endl;
    itemListBuffer.updateBuffer(itemBuffer.itemList.data(), itemListSize, 0);

    infoBuffer.bind(LIGHT_CLUSTER_INFO_BINDING_POINT);
    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);
    stopTimer(1);
    assert_no_glerror();
}

void CPUPlaneClusterer::buildClusters(Camera* cam)
{
    // FIXME Remove:
    depthSplits = 4;

    clustersDirty = false;
    float camNear = cam->zNear;
    float camFar  = cam->zFar;
    cam->recomputeProj();
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
        gridCount[2] = depthSplits;
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
    clusterCache.clear();
    clusterCache.resize(clusterCount);

    planesX.clear();
    planesY.clear();
    planesZ.clear();
    planesX.resize((int)gridCount[0] + 1);
    planesY.resize((int)gridCount[1] + 1);
    planesZ.resize((int)gridCount[2] + 1);
    if (renderDebugEnabled && debugFrustumToView)
    {
        debugCluster.lines.clear();
    }

    // const vec3 eyeView(0.0); Not required because it is zero.
    PointVertex v;

    v.color = vec3(0.5, 0.125, 0);

    vec3 p0, p1, p2;

    for (int x = 0; x < planesX.size(); ++x)
    {
        vec4 screenSpaceBL(x * screenSpaceTileSize, 0, -1.0, 1.0);       // Bottom left
        vec4 screenSpaceTL(x * screenSpaceTileSize, height, -1.0, 1.0);  // Top left

        vec3 viewBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
        vec3 viewTL(make_vec3(viewPosFromScreenPos(screenSpaceTL, invProjection)));

        vec3 viewNearPlaneBL(zeroZIntersection(viewBL, -camNear));
        vec3 viewNearPlaneTL(zeroZIntersection(viewTL, -camNear));

        vec3 viewFarPlaneBL(zeroZIntersection(viewBL, -camFar));
        vec3 viewFarPlaneTL(zeroZIntersection(viewTL, -camFar));

        p0 = viewNearPlaneBL;
        p1 = viewFarPlaneBL;
        p2 = viewFarPlaneTL;

        planesX[x] = Plane(p0, p1, p2);

        if (renderDebugEnabled && debugFrustumToView)
        {
            v.position = viewFarPlaneTL;
            debugCluster.lines.push_back(v);
            v.position = viewFarPlaneTL + planesX[x].normal;
            v.color    = vec3(1, 0, 0);
            debugCluster.lines.push_back(v);
            v.color = vec3(0.5, 0.125, 0);

            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneTL;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewFarPlaneTL;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewFarPlaneBL;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
        }
    }

    for (int y = 0; y < planesY.size(); ++y)
    {
        vec4 screenSpaceBL(0, y * screenSpaceTileSize, -1.0, 1.0);      // Bottom left
        vec4 screenSpaceBR(width, y * screenSpaceTileSize, -1.0, 1.0);  // Bottom Right

        vec3 viewBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
        vec3 viewBR(make_vec3(viewPosFromScreenPos(screenSpaceBR, invProjection)));

        vec3 viewNearPlaneBL(zeroZIntersection(viewBL, -camNear));
        vec3 viewNearPlaneBR(zeroZIntersection(viewBR, -camNear));

        vec3 viewFarPlaneBL(zeroZIntersection(viewBL, -camFar));
        vec3 viewFarPlaneBR(zeroZIntersection(viewBR, -camFar));

        p0 = viewNearPlaneBR;
        p1 = viewFarPlaneBR;
        p2 = viewFarPlaneBL;

        planesY[y] = Plane(p0, p1, p2);

        if (renderDebugEnabled && debugFrustumToView)
        {
            v.position = viewFarPlaneBL;
            debugCluster.lines.push_back(v);
            v.position = viewFarPlaneBL + planesY[y].normal;
            v.color    = vec3(1, 0, 0);
            debugCluster.lines.push_back(v);
            v.color = vec3(0.5, 0.125, 0);

            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneBR;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewFarPlaneBR;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewFarPlaneBL;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
        }
    }

    int numZPlanes = gridCount[2];
    for (int z = 0; z < planesZ.size(); ++z)
    {
        vec4 screenSpaceC(width / 2, height / 2, -1.0, 1.0);  // Center Point

        vec3 viewNearPlaneC(make_vec3(viewPosFromScreenPos(screenSpaceC, invProjection)));

        // Doom Depth Split, because it looks good.
        // float tileNear = -camNear * pow(camFar / camNear, (float)z / gridCount[2]);
        float tileFar = -camNear * pow(camFar / camNear, (float)(numZPlanes - z) / gridCount[2]);

        vec3 viewFarClusterC(zeroZIntersection(viewNearPlaneC, tileFar));

        planesZ[z] = Plane(viewFarClusterC, vec3(0, 0, 1));

        if (renderDebugEnabled && debugFrustumToView)
        {
            vec4 screenSpaceBL(0, 0, -1.0, 1.0);           // Bottom left
            vec4 screenSpaceTL(0, height, -1.0, 1.0);      // Top left
            vec4 screenSpaceBR(width, 0, -1.0, 1.0);       // Bottom Right
            vec4 screenSpaceTR(width, height, -1.0, 1.0);  // Top Right

            vec3 viewBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, invProjection)));
            vec3 viewTL(make_vec3(viewPosFromScreenPos(screenSpaceTL, invProjection)));
            vec3 viewBR(make_vec3(viewPosFromScreenPos(screenSpaceBR, invProjection)));
            vec3 viewTR(make_vec3(viewPosFromScreenPos(screenSpaceTR, invProjection)));

            vec3 viewNearPlaneBL(zeroZIntersection(viewBL, tileFar));
            vec3 viewNearPlaneTL(zeroZIntersection(viewTL, tileFar));
            vec3 viewNearPlaneBR(zeroZIntersection(viewBR, tileFar));
            vec3 viewNearPlaneTR(zeroZIntersection(viewTR, tileFar));

            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneBL + planesZ[z].normal;
            v.color    = vec3(1, 0, 0);
            debugCluster.lines.push_back(v);
            v.color = vec3(0.5, 0.125, 0);

            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneTL;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneTR;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneBR;
            debugCluster.lines.push_back(v);
            debugCluster.lines.push_back(v);
            v.position = viewNearPlaneBL;
            debugCluster.lines.push_back(v);
        }
    }

    if (renderDebugEnabled && debugFrustumToView)
    {
        debugPoints.resize(clusterCount);
        for (int x = 0; x < (int)gridCount[0]; ++x)
        {
            for (int y = 0; y < (int)gridCount[1]; ++y)
            {
                for (int z = 0; z < (int)gridCount[2]; ++z)
                {
                    vec4 screenSpaceBL(x * screenSpaceTileSize, y * screenSpaceTileSize, -1.0, 1.0);  // Bottom left
                    vec4 screenSpaceTL(x * screenSpaceTileSize, (y + 1) * screenSpaceTileSize, -1.0, 1.0);  // Top left
                    vec4 screenSpaceBR((x + 1) * screenSpaceTileSize, y * screenSpaceTileSize, -1.0,
                                       1.0);  // Bottom Right
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

                    clusterDebugPoints dbg;
                    dbg.nBL = viewNearClusterBL;
                    dbg.nBR = viewNearClusterBR;
                    dbg.nTL = viewNearClusterTL;
                    dbg.nTR = viewNearClusterTR;
                    dbg.fBL = viewFarClusterBL;
                    dbg.fBR = viewFarClusterBR;
                    dbg.fTL = viewFarClusterTL;
                    dbg.fTR = viewFarClusterTR;

                    int tileIndex = x + (int)gridCount[0] * y + (int)(gridCount[0] * gridCount[1]) * z;

                    debugPoints.at(tileIndex) = dbg;
                }
            }
        }
    }

    if (renderDebugEnabled && debugFrustumToView)
    {
        debugCluster.lineWidth = 1;

        debugCluster.setModelMatrix(cam->getModelMatrix());  // is inverse view.
        debugCluster.translateLocal(vec3(0, 0, -0.0001f));
#if 0
        debugCluster.setPosition(make_vec4(0));
        debugCluster.translateGlobal(vec3(0, 6, 0));
        debugCluster.setScale(make_vec3(0.33f));
#endif
        debugCluster.calculateModel();
        debugCluster.updateBuffer();
    }

    startTimer(0);
    clusterInfoBuffer.tileDebug = tileDebugView ? avgAllowedItemsPerCluster : 0;

    itemBuffer.itemList.clear();
    itemBuffer.itemList.resize(avgAllowedItemsPerCluster * clusterInfoBuffer.clusterListCount);
    clusterInfoBuffer.itemListCount = itemBuffer.itemList.size();

    int itemBufferSize = sizeof(itemBuffer) + sizeof(clusterItem) * itemBuffer.itemList.size();
    int maxBlockSize   = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
    SAIGA_ASSERT(maxBlockSize > itemBufferSize, "Item SSB size too big!");

    itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), itemBufferSize, GL_DYNAMIC_DRAW);

    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.createGLBuffer(clusterBuffer.clusterList.data(), clusterListSize, GL_DYNAMIC_DRAW);

    infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);

    stopTimer(0);
}

bool CPUPlaneClusterer::fillImGui()
{
    bool changed = Clusterer::fillImGui();

    changed |= ImGui::Checkbox("refinement", &refinement);

    ImGui::Text("avgAllowedItemsPerCluster: %d", avgAllowedItemsPerCluster);

    ImGui::Text("ItemListByteSize: %d", itemBuffer.itemList.size() * sizeof(clusterItem));

    return changed;
}

}  // namespace Saiga
