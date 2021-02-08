/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/rendering/lighting/light_clusterer.h"

#include "saiga/core/imgui/imgui.h"


namespace Saiga
{
Clusterer::Clusterer(ClustererParameters _params)
{
    clusterThreeDimensional = _params.clusterThreeDimensional;
    useTimers               = _params.useTimers;
    clustersDirty           = true;

    infoBuffer.createGLBuffer(nullptr, sizeof(infoBuf_t), GL_DYNAMIC_DRAW);
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
}

void Clusterer::loadComputeShaders() {}

void Clusterer::clusterLights(Camera* cam, const ViewPort& viewPort)
{
    assert_no_glerror();
    float current_depth_range = cam->zFar - cam->zNear;
    if (clusterThreeDimensional && depth != current_depth_range) clustersDirty = true;
    depth = current_depth_range;

    if (clustersDirty) build_clusters(cam);

    lightAssignmentTimer.start();

    // memset(itemBuffer.itemList.data(), 0, sizeof(clusterItem) * itemBuffer.itemList.size());
    const int maxClusterItemsPerCluster = 256;  // TODO Paul: Hardcoded?

    std::vector<vec4> view_space_lights(pointLightsClusterData.size());

    PointVertex v;
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
            vec4& plc         = view_space_lights[i];
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
            }
        }

        cluster& gpuCluster = clusterBuffer.clusterList.at(c);
        gpuCluster.offset   = c * maxClusterItemsPerCluster;

        int count = 0;
        for (int i = 0; i < registered.size(); ++i)
        {
            int& idx = registered.at(i);
            if (idx < 0) continue;
            if (count >= maxClusterItemsPerCluster)  // TODO Paul...
                break;
            itemBuffer.itemList.at(gpuCluster.offset + count++).plIdx = idx;
        }

        gpuCluster.plCount = count;
        gpuCluster.slCount = 0;
        gpuCluster.blCount = 0;
    }

#pragma omp barrier
    lightAssignmentTimer.stop();

    startTimer(1);

    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.updateBuffer(clusterBuffer.clusterList.data(), clusterListSize, 0);

    int itemListSize = sizeof(clusterItem) * itemBuffer.itemList.size();
    itemListBuffer.updateBuffer(itemBuffer.itemList.data(), itemListSize, 0);

    infoBuffer.bind(LIGHT_CLUSTER_INFO_BINDING_POINT);
    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);
    stopTimer(1);
    assert_no_glerror();
}

void Clusterer::build_clusters(Camera* cam)
{
    clustersDirty              = false;
    clusterInfoBuffer.clusterX = splitX;
    clusterInfoBuffer.clusterY = splitY;
    clusterInfoBuffer.clusterZ = splitZ;

    clusterInfoBuffer.screenWidth  = width;
    clusterInfoBuffer.screenHeight = height;
    clusterInfoBuffer.zNear        = cam->zNear;
    clusterInfoBuffer.zFar         = cam->zFar;
    clusterInfoBuffer.scale        = (float)(splitZ) / log2(cam->zFar / cam->zNear);
    clusterInfoBuffer.bias         = -((float)splitZ * log2(cam->zNear) / log2(cam->zFar / cam->zNear));


    // Calculate Cluster Planes in View Space.
    int clusterCount = splitX * splitY * splitZ;
    clusterBuffer.clusterList.clear();
    clusterBuffer.clusterList.resize(clusterCount);
    clusterInfoBuffer.clusterListCount = clusterCount;

    int tileWidth  = std::ceil((float)width / (float)splitX);
    int tileHeight = std::ceil((float)height / (float)splitY);


    culling_cluster.clear();
    culling_cluster.resize(clusterCount);
    if (renderDebugEnabled && debugFrustumToView) debugCluster.lines.clear();

    // const vec3 eyeView(0.0); Not required because it is zero.

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

                if (renderDebugEnabled && debugFrustumToView)
                {
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
                }

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
        debugFrustumToView = false;
    }

    startTimer(0);
    itemBuffer.itemList.clear();
    int maxClusterItemsPerCluster = 256;  // TODO Paul: Hardcoded?
    itemBuffer.itemList.resize(maxClusterItemsPerCluster * culling_cluster.size());
    clusterInfoBuffer.itemListCount = itemBuffer.itemList.size();

    int itemBufferSize = sizeof(itemBuffer) + sizeof(clusterItem) * itemBuffer.itemList.size();
    int maxBlockSize   = ShaderStorageBuffer::getMaxShaderStorageBlockSize();
    SAIGA_ASSERT(maxBlockSize < itemBufferSize, "Item SSB size too big!");

    itemListBuffer.createGLBuffer(itemBuffer.itemList.data(), itemBufferSize, GL_DYNAMIC_DRAW);

    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.createGLBuffer(clusterBuffer.clusterList.data(), clusterListSize, GL_DYNAMIC_DRAW);

    infoBuffer.updateBuffer(&clusterInfoBuffer, sizeof(clusterInfoBuffer), 0);

    stopTimer(0);
}


void Clusterer::renderImGui(bool* p_open)
{
    ImGui::Begin("Clusterer", p_open);

    ImGui::Text("resolution: %dx%d", width, height);
    if (ImGui::Checkbox("renderDebugEnabled", &renderDebugEnabled) && renderDebugEnabled)
    {
        clustersDirty      = true;
        debugFrustumToView = true;
    }
    if (renderDebugEnabled)
        if (ImGui::Button("debugFrustumToView")) debugFrustumToView = true;

    if (ImGui::Checkbox("useTimers", &useTimers) && useTimers)
    {
        gpuTimers.resize(2);
        gpuTimers[0].create();
        gpuTimers[1].create();
        timerStrings.resize(2);
        timerStrings[0] = "Rebuilding Clusters";
        timerStrings[1] = "Light Assignment Buffer Update";
        lightAssignmentTimer.stop();
    }


    if (useTimers)
    {
        ImGui::Text("Render Time (without shadow map computation)");
        for (int i = 0; i < 2; ++i)
        {
            ImGui::Text("  %f ms %s", getTime(i), timerStrings[i].c_str());
        }
        ImGui::Text("  %f ms %s", lightAssignmentTimer.getTimeMS(), "CPU Light Assignment");
    }
    ImGui::Checkbox("clusterThreeDimensional", &clusterThreeDimensional);
    bool changed = false;
    changed |= ImGui::SliderInt("splitX", &splitX, 1, 32);
    changed |= ImGui::SliderInt("splitY", &splitY, 1, 32);
    if (clusterThreeDimensional)
    {
        changed |= ImGui::SliderInt("splitZ", &splitZ, 1, 32);
    }
    else
        splitZ = 1;


    clustersDirty |= changed | debugFrustumToView;

    ImGui::End();
}
}  // namespace Saiga
