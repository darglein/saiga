/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/light_clusterer.h"
namespace Saiga
{
#define MIN(X, Y) ((X < Y) ? X : Y)
#define MAX(X, Y) ((X > Y) ? X : Y)
#define MINV(V1, V2) vec3(MIN(V1[0], V2[0]), MIN(V1[1], V2[1]), MIN(V1[2], V2[2]))
#define MAXV(V1, V2) vec3(MAX(V1[0], V2[0]), MAX(V1[1], V2[1]), MAX(V1[2], V2[2]))

Clusterer::Clusterer(ClustererParameters _params)
{
    clusterThreeDimensional = _params.clusterThreeDimensional;
    useTimers               = _params.useTimers;
    clustersDirty           = true;

    clusterListBuffer.createGLBuffer(nullptr, 0, GL_DYNAMIC_DRAW);
    itemListBuffer.createGLBuffer(nullptr, 0, GL_DYNAMIC_DRAW);

    loadComputeShaders();
}

Clusterer::~Clusterer() {}

void Clusterer::init(int _width, int _height, bool _useTimers)
{
    width         = _width;
    height        = _height;
    useTimers     = _useTimers;
    clustersDirty = true;

    splitX = _width / 100;
    splitY = _height / 100;

    if (clusterThreeDimensional) splitZ = 8;  // TODO Paul!

    if (useTimers)
    {
        timers2.resize(2);
        timers2[0].create();
        timers2[1].create();
        timerStrings.resize(2);
        timerStrings[0] = "Rebuilding Clusters";
        timerStrings[1] = "Light Assignment";
    }
}

void Clusterer::resize(int _width, int _height)
{
    width         = _width;
    height        = _height;
    clustersDirty = true;

    splitX = _width / 100;
    splitY = _height / 100;

    if (clusterThreeDimensional) splitZ = 8;  // TODO Paul!
}

void Clusterer::loadComputeShaders() {}

void Clusterer::clusterLights(Camera* cam, const ViewPort& viewPort)
{
    float current_depth_range = cam->zFar - cam->zNear;
    if (clusterThreeDimensional && depth != current_depth_range) clustersDirty = true;
    depth = current_depth_range;

    if (clustersDirty) build_clusters(cam);

    startTimer(1);

    int realItemListSize = 0;

    itemBuffer.itemList.clear();
    int maxClusterItemsPerCluster = 64;  // TODO Paul: Hardcoded?
    itemBuffer.itemList.resize(maxClusterItemsPerCluster * culling_cluster.size());

    int PlsInCluster   = 0;
    int SlsInCluster   = 0;
    int BlsInCluster   = 0;
    int itemsInCluster = 0;
    int plIdx          = 0;

    for (long c = 0; c < culling_cluster.size(); ++c)
    {
        PlsInCluster   = 0;
        SlsInCluster   = 0;
        BlsInCluster   = 0;
        itemsInCluster = 0;
        plIdx          = 0;

        const AABB& clusterAABB = culling_cluster[c].bounds;
        for (PointLightClusterData& plc : pointLightsClusterData)
        {
            vec3 view_c(cam->projectToViewSpace(plc.world_center));  // These seeam to cost the most ... But maybe the
                                                                     // AABBs are just completely broken ...
            if (Intersection::SphereAABB(view_c, plc.radius,
                                         clusterAABB))  // These seeam to cost the most ... But maybe the AABBs are just
                                                        // completely broken ...
            {
                if (PlsInCluster >= maxClusterItemsPerCluster) break;  // TODO Paul...
                itemBuffer.itemList.at(realItemListSize + PlsInCluster).plIdx = plIdx;
                PlsInCluster++;
            }
            plIdx++;
        }

        itemsInCluster = std::max(std::max(PlsInCluster, SlsInCluster), BlsInCluster);

        cluster& gpuCluster = clusterBuffer.clusterList.at(c);
        gpuCluster.offset   = realItemListSize;
        gpuCluster.plCount  = PlsInCluster;
        gpuCluster.slCount  = SlsInCluster;
        gpuCluster.blCount  = BlsInCluster;

        realItemListSize += itemsInCluster;
    }

    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);

    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.updateBuffer(clusterBuffer.clusterList.data(), clusterListSize, offsetof(clusterBuffer_t, p0));

    itemListBuffer.updateBuffer(itemBuffer.itemList.data(), sizeof(clusterItem) * realItemListSize, 0);
    stopTimer(1);
    // For now:
    static int frame_delim = 0;
    if (frame_delim % 30 == 0) printTimings();
    frame_delim++;
}

void Clusterer::build_clusters(Camera* cam)
{
    startTimer(0);
    clustersDirty          = false;
    clusterBuffer.clusterX = splitX;
    clusterBuffer.clusterY = splitY;
    clusterBuffer.clusterZ = splitZ;

    clusterBuffer.screenWidth  = width;
    clusterBuffer.screenHeight = height;

    // Calculate Cluster Planes in View Space.
    int clusterCount = splitX * splitY * splitZ;
    clusterBuffer.clusterList.clear();
    clusterBuffer.clusterList.resize(clusterCount);
    int clusterBufferSize = sizeof(clusterBuffer) + sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.createGLBuffer(&clusterBuffer, clusterBufferSize, GL_DYNAMIC_DRAW);

    int tileWidth  = width / splitX;
    int tileHeight = height / splitY;


    culling_cluster.clear();
    culling_cluster.resize(clusterCount);

    // const vec3 eyeView(0.0); Not required because it is zero.

    // Using AABB. This is not perfectly exact, but good enough and faster.

    for (int x = 0; x < splitX; ++x)
    {
        for (int y = 0; y < splitY; ++y)
        {
            for (int z = 0; z < splitZ; ++z)
            {
                vec4 screenSpaceMin(x * tileWidth, y * tileHeight, -1.0, 1.0);              // Bottom left
                vec4 screenSpaceMax((x + 1) * tileWidth, (y + 1) * tileHeight, -1.0, 1.0);  // Top Right

                float camNear = cam->zNear;
                float camFar  = cam->zFar;

                // Doom Depth Split, because it looks good.
                float tileNear = -camNear * pow(camFar / camNear, (float)z / (float)splitZ);
                float tileFar  = -camNear * pow(camFar / camNear, (float)(z + 1) / (float)splitZ);

                vec3 viewMin(make_vec3(viewPosFromScreenPos(screenSpaceMin, cam)));
                vec3 viewMax(make_vec3(viewPosFromScreenPos(screenSpaceMax, cam)));

                vec3 minNear(zeroZIntersection(viewMin, tileNear));
                vec3 minFar(zeroZIntersection(viewMin, tileFar));
                vec3 maxNear(zeroZIntersection(viewMax, tileNear));
                vec3 maxFar(zeroZIntersection(viewMax, tileFar));

                vec3 minMin = MINV(minNear, minFar);
                vec3 minMax = MINV(maxNear, maxFar);
                vec3 maxMin = MAXV(minNear, minFar);
                vec3 maxMax = MAXV(maxNear, maxFar);

                vec3 AABBmin = MINV(minMin, minMax);
                vec3 AABBmax = MAXV(maxMin, maxMax);

                int tileIndex = x + splitX * y + (splitX * splitY) * z;

                culling_cluster.at(tileIndex).bounds.min = AABBmin;
                culling_cluster.at(tileIndex).bounds.max = AABBmax;
            }
        }
    }

    itemBuffer.itemList.clear();
    int maxClusterItemsPerCluster = 64;  // TODO Paul: Hardcoded?
    itemBuffer.itemList.resize(maxClusterItemsPerCluster * culling_cluster.size());
    itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), sizeof(clusterItem) * itemBuffer.itemList.size(),
                                  GL_DYNAMIC_DRAW);

    stopTimer(0);
}
}  // namespace Saiga
