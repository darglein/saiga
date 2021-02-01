/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/light_clusterer.h"

//#define DEBUG_DRAW
#define DEBUG_IN_SCREEN_SPACE

namespace Saiga
{

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

    // splitX = 16;
    // splitY = 8;
    //
    // if (clusterThreeDimensional) splitZ = 8;  // TODO Paul!

    if (useTimers)
    {
        gpuTimers.resize(2);
        gpuTimers[0].create();
        gpuTimers[1].create();
        timerStrings.resize(2);
        timerStrings[0] = "Rebuilding Clusters";
        timerStrings[1] = "Light Assignment Buffer Update";
        lightAssignmentTimer.stop();
    }
}

void Clusterer::resize(int _width, int _height)
{
    width         = _width;
    height        = _height;
    clustersDirty = true;

    // splitX = 16;
    // splitY = 8;
    //
    // if (clusterThreeDimensional) splitZ = 8;  // TODO Paul!
}

void Clusterer::loadComputeShaders() {}

void Clusterer::clusterLights(Camera* cam, const ViewPort& viewPort)
{
    float current_depth_range = cam->zFar - cam->zNear;
    if (clusterThreeDimensional && depth != current_depth_range) clustersDirty = true;
    depth = current_depth_range;

    // startTimer(0);
    if (clustersDirty) build_clusters(cam);
        // stopTimer(0);

#ifdef DEBUG_DRAW
    static int lastSize = pointLightsClusterData.size();

    if (lastSize == pointLightsClusterData.size()) return;
    lastSize = pointLightsClusterData.size();

    debugLights.points.clear();
    debugLights.pointSize = 6;
#endif

    lightAssignmentTimer.start();

    // memset(itemBuffer.itemList.data(), 0, sizeof(clusterItem) * itemBuffer.itemList.size());
    const int maxClusterItemsPerCluster = 256;  // TODO Paul: Hardcoded?

    std::vector<vec4> view_space_lights(pointLightsClusterData.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pointLightsClusterData.size(); ++i)
    {
        PointLightClusterData& plc = pointLightsClusterData[i];
        view_space_lights[i]       = make_vec4(cam->projectToViewSpace(plc.world_center), plc.radius);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < culling_cluster.size(); ++c)
    {
        const auto& cluster_planes = culling_cluster[c].planes;
        std::vector<int> registered(view_space_lights.size(), -1);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < view_space_lights.size(); ++i)
        {
            vec4& plc = view_space_lights[i];
            PointVertex v;
            bool intersection = true;
            float distance;
            vec3 sphereCenter  = make_vec3(plc);
            float sphereRadius = plc.w();
            for (int i = 0; i < 6; ++i)
            {
                distance = cluster_planes[i].distance(sphereCenter);
                if (distance < -sphereRadius)
                {
                    intersection = false;
                    break;
                }
            }
            if (intersection)
            {
                registered.at(i) = i;
#ifdef DEBUG_DRAW
            v.color           = vec3(1, 1, 0);
            v.position = sphereCenter;
            debugLights.points.push_back(v);
#endif
            }

        }

        cluster& gpuCluster = clusterBuffer.clusterList.at(c);
        gpuCluster.offset   = c * maxClusterItemsPerCluster;

        int count = 0;
        for (int i = 0; i < registered.size(); ++i)
        {
            int& idx = registered.at(i);
            if (idx < 0) continue;
            if (i >= maxClusterItemsPerCluster) break;  // TODO Paul...
            itemBuffer.itemList.at(gpuCluster.offset + count++).plIdx = idx;
        }

        gpuCluster.plCount = count;
        gpuCluster.slCount = 0;
        gpuCluster.blCount = 0;
    }

#ifdef DEBUG_DRAW
#    ifdef DEBUG_IN_SCREEN_SPACE
    debugLights.setModelMatrix(cam->getModelMatrix());  // is inverse view.
    debugLights.translateLocal(vec3(0, 0, -0.0001f));
#    else
    debugLights.setPosition(make_vec4(0));
    debugLights.translateGlobal(vec3(0, 6, 0));
    debugLights.setScale(make_vec3(0.33f));
#    endif
    debugLights.calculateModel();
    debugLights.updateBuffer();
#endif

    lightAssignmentTimer.stop();

    // startTimer(1);
    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);

    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.updateBuffer(clusterBuffer.clusterList.data(), clusterListSize, offsetof(clusterBuffer_t, p0));

    itemListBuffer.updateBuffer(itemBuffer.itemList.data(), sizeof(clusterItem) * itemBuffer.itemList.size(), 0);
    // stopTimer(1);
    // For now:
    static int frame_delim = 0;
    if (frame_delim % 30 == 0) printTimings();
    frame_delim++;
}

void Clusterer::build_clusters(Camera* cam)
{
    clustersDirty          = false;
    clusterBuffer.clusterX = splitX;
    clusterBuffer.clusterY = splitY;
    clusterBuffer.clusterZ = splitZ;

    clusterBuffer.screenWidth  = width;
    clusterBuffer.screenHeight = height;
    clusterBuffer.zNear = cam->zNear;
    clusterBuffer.zFar = cam->zFar;
    clusterBuffer.scale = (float)(splitZ) / log2(cam->zFar /cam->zNear);
    clusterBuffer.bias = ((float)splitZ * log2(cam->zNear) / log2(cam->zFar / cam->zNear));

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
    debugCluster.lines.clear();

    // const vec3 eyeView(0.0); Not required because it is zero.

    // Using AABB. This is not perfectly exact, but good enough and faster.

    for (int x = 0; x < splitX; ++x)
    {
        for (int y = 0; y < splitY; ++y)
        {
            for (int z = 0; z < splitZ; ++z)
            {
                vec4 screenSpaceBL(x * tileWidth, y * tileHeight, -1.0, 1.0);              // Bottom left
                vec4 screenSpaceTL(x * tileWidth, (y + 1) * tileHeight, -1.0, 1.0);        // Top left
                vec4 screenSpaceBR((x + 1) * tileWidth, y * tileHeight, -1.0, 1.0);        // Bottom Right
                vec4 screenSpaceTR((x + 1) * tileWidth, (y + 1) * tileHeight, -1.0, 1.0);  // Top Right

                float camNear = cam->zNear;
                float camFar  = cam->zFar;

                // Doom Depth Split, because it looks good.
                float tileNear = -camNear * pow(camFar / camNear, (float)z / (float)splitZ);
                float tileFar  = -camNear * pow(camFar / camNear, (float)(z + 1) / (float)splitZ);

                vec3 viewNearPlaneBL(make_vec3(viewPosFromScreenPos(screenSpaceBL, cam)));
                vec3 viewNearPlaneTL(make_vec3(viewPosFromScreenPos(screenSpaceTL, cam)));
                vec3 viewNearPlaneBR(make_vec3(viewPosFromScreenPos(screenSpaceBR, cam)));
                vec3 viewNearPlaneTR(make_vec3(viewPosFromScreenPos(screenSpaceTR, cam)));

                vec3 viewNearClusterBL(zeroZIntersection(viewNearPlaneBL, tileNear));
                vec3 viewNearClusterTL(zeroZIntersection(viewNearPlaneTL, tileNear));
                vec3 viewNearClusterBR(zeroZIntersection(viewNearPlaneBR, tileNear));
                vec3 viewNearClusterTR(zeroZIntersection(viewNearPlaneTR, tileNear));

                vec3 viewFarClusterBL(zeroZIntersection(viewNearPlaneBL, tileFar));
                vec3 viewFarClusterTL(zeroZIntersection(viewNearPlaneTL, tileFar));
                vec3 viewFarClusterBR(zeroZIntersection(viewNearPlaneBR, tileFar));
                vec3 viewFarClusterTR(zeroZIntersection(viewNearPlaneTR, tileFar));

#ifdef DEBUG_DRAW
                PointVertex v;
                v.color = vec3(1, 0.25, 0);

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
#endif

                vec3 p0, p1, p2;

                p0                 = viewNearClusterBL;
                p1                 = viewFarClusterBL;
                p2                 = viewFarClusterTL;
                vec3 nLeftPlane    = normalize(cross(p1 - p0, p2 - p0));
                float offLeftPlane = dot(p1, nLeftPlane);

                p0                = viewNearClusterTL;
                p1                = viewFarClusterTL;
                p2                = viewFarClusterTR;
                vec3 nTopPlane    = normalize(cross(p1 - p0, p2 - p0));
                float offTopPlane = dot(p1, nTopPlane);

                p0                  = viewNearClusterTR;
                p1                  = viewFarClusterTR;
                p2                  = viewFarClusterBR;
                vec3 nRightPlane    = normalize(cross(p1 - p0, p2 - p0));
                float offRightPlane = dot(p1, nRightPlane);

                p0                   = viewNearClusterBR;
                p1                   = viewFarClusterBR;
                p2                   = viewFarClusterBL;
                vec3 nBottomPlane    = normalize(cross(p1 - p0, p2 - p0));
                float offBottomPlane = dot(p1, nBottomPlane);

                vec3 nBackPlane    = vec3(0, 0, 1);
                float offBackPlane = dot(viewFarClusterTR, nBackPlane);

                vec3 nFrontPlane    = vec3(0, 0, -1);
                float offFrontPlane = dot(viewNearClusterBL, nFrontPlane);

                int tileIndex = x + splitX * y + (splitX * splitY) * z;

                auto& planes     = culling_cluster.at(tileIndex).planes;
                planes[0].normal = nLeftPlane;
                planes[0].d      = offLeftPlane;
                planes[1].normal = nTopPlane;
                planes[1].d      = offTopPlane;
                planes[2].normal = nRightPlane;
                planes[2].d      = offRightPlane;
                planes[3].normal = nBottomPlane;
                planes[3].d      = offBottomPlane;
                planes[4].normal = nBackPlane;
                planes[4].d      = offBackPlane;
                planes[5].normal = nFrontPlane;
                planes[5].d      = offFrontPlane;
            }
        }
    }

#ifdef DEBUG_DRAW
    debugCluster.lineWidth = 1;
#    ifdef DEBUG_IN_SCREEN_SPACE
    debugCluster.setModelMatrix(cam->getModelMatrix());  // is inverse view.
    debugCluster.translateLocal(vec3(0, 0, -0.0001f));
#    else
    debugCluster.setPosition(make_vec4(0));
    debugCluster.translateGlobal(vec3(0, 6, 0));
    debugCluster.setScale(make_vec3(0.33f));
#    endif
    debugCluster.calculateModel();
    debugCluster.updateBuffer();
#endif

    itemBuffer.itemList.clear();
    int maxClusterItemsPerCluster = 256;  // TODO Paul: Hardcoded?
    itemBuffer.itemList.resize(maxClusterItemsPerCluster * culling_cluster.size());
    int maxBlockSize = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
    itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), sizeof(clusterItem) * itemBuffer.itemList.size(),
                                  GL_DYNAMIC_DRAW);
}
}  // namespace Saiga
