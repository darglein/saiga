/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

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
    // TODO Paul: Dirty Clusters on FOV and Aspect change usw...
    // clustersDirty |= depth != cam->zFar - cam->zNear;
    // depth = cam->zFar - cam->zNear;

    if (clustersDirty) build_clusters(cam);

    lightAssignmentTimer.start();

    // memset(itemBuffer.itemList.data(), 0, sizeof(clusterItem) * itemBuffer.itemList.size());
    const int maxClusterItemsPerCluster = 256;  // TODO Paul: Hardcoded?
    int visibleLightIndices[maxClusterItemsPerCluster];

    int globalOffset = 0;

    for (int c = 0; c < culling_cluster.size(); ++c)
    {
        const auto& cluster_planes = culling_cluster[c].planes;

        int visibleLightCount = 0;

        for (int i = 0; i < pointLightsClusterData.size(); ++i)
        {
            PointLightClusterData& plc = pointLightsClusterData[i];
            bool intersection          = true;
            float pDotC;
            vec3 sphereCenter  = cam->projectToViewSpace(plc.world_center);
            float sphereRadius = plc.radius;
            for (int p = 0; p < 6; ++p)
            {
                pDotC = dot(cluster_planes[p].normal, sphereCenter) - cluster_planes[p].d + sphereRadius;

                if (pDotC < 0.0)
                {
                    intersection = false;
                    break;
                }
            }
            if (intersection)
            {
                if (visibleLightCount >= maxClusterItemsPerCluster)  // TODO Paul...
                    break;

                visibleLightIndices[visibleLightCount] = i;
                visibleLightCount++;
            }
        }


        cluster& gpuCluster = clusterBuffer.clusterList.at(c);
        gpuCluster.offset   = globalOffset;
        globalOffset += visibleLightCount;

        for (int v = 0; v < visibleLightCount; ++v)
        {
            itemBuffer.itemList.at(gpuCluster.offset + v).plIdx = visibleLightIndices[v];
        }

        gpuCluster.plCount = visibleLightCount;
        gpuCluster.slCount = 0;
        gpuCluster.blCount = 0;
    }

    lightAssignmentTimer.stop();

    startTimer(1);

    int clusterListSize = sizeof(cluster) * clusterBuffer.clusterList.size();
    clusterListBuffer.updateBuffer(clusterBuffer.clusterList.data(), clusterListSize, 0);

    int itemListSize = sizeof(clusterItem) * itemBuffer.itemList.size();
    // std::cout << "Used " << globalOffset * sizeof(clusterItem) << " item slots of " << itemListSize << std::endl;
    itemListBuffer.updateBuffer(itemBuffer.itemList.data(), itemListSize, 0);

    infoBuffer.bind(LIGHT_CLUSTER_INFO_BINDING_POINT);
    clusterListBuffer.bind(LIGHT_CLUSTER_LIST_BINDING_POINT);
    itemListBuffer.bind(LIGHT_CLUSTER_ITEM_LIST_BINDING_POINT);
    stopTimer(1);
    assert_no_glerror();
}

void Clusterer::build_clusters(Camera* cam)
{
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

    culling_cluster.clear();
    culling_cluster.resize(clusterCount);
    if (renderDebugEnabled && debugFrustumToView)
    {
        debugCluster.lines.clear();
        debugPoints.points.clear();
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


                if (renderDebugEnabled && debugFrustumToView)
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

                    /*
                                        vec3 center;

                                        center     = (viewNearClusterBL + viewNearClusterTL + viewFarClusterBL +
                       viewFarClusterTL) * 0.25; v.color    = vec3(1, 0, 0); v.position = center;
                                        debugCluster.lines.push_back(v);
                                        v.color    = vec3(0, 1, 0);
                                        v.position = center + planes[0].normal * 1.0;
                                        debugCluster.lines.push_back(v);

                                        center     = (viewNearClusterTL + viewNearClusterTR + viewFarClusterTL +
                       viewFarClusterTR) * 0.25; v.color    = vec3(1, 0, 0); v.position = center;
                                        debugCluster.lines.push_back(v);
                                        v.color    = vec3(0, 1, 0);
                                        v.position = center + planes[1].normal * 1.0;
                                        debugCluster.lines.push_back(v);

                                        center     = (viewNearClusterBR + viewNearClusterTR + viewFarClusterBR +
                       viewFarClusterTR) * 0.25; v.color    = vec3(1, 0, 0); v.position = center;
                                        debugCluster.lines.push_back(v);
                                        v.color    = vec3(0, 1, 0);
                                        v.position = center + planes[2].normal * 1.0;
                                        debugCluster.lines.push_back(v);

                                        center     = (viewNearClusterBL + viewNearClusterBR + viewFarClusterBL +
                       viewFarClusterBR) * 0.25; v.color    = vec3(1, 0, 0); v.position = center;
                                        debugCluster.lines.push_back(v);
                                        v.color    = vec3(0, 1, 0);
                                        v.position = center + planes[3].normal * 1.0;
                                        debugCluster.lines.push_back(v);

                                        center     = (viewFarClusterTL + viewFarClusterTR + viewFarClusterBL +
                       viewFarClusterBR) * 0.25; v.color    = vec3(1, 0, 0); v.position = center;
                                        debugCluster.lines.push_back(v);
                                        v.color    = vec3(0, 1, 0);
                                        v.position = center + planes[4].normal * 1.0;
                                        debugCluster.lines.push_back(v);

                                        center     = (viewNearClusterTL + viewNearClusterTR + viewNearClusterBL +
                       viewNearClusterBR) * 0.25; v.color    = vec3(1, 0, 0); v.position = center;
                                        debugCluster.lines.push_back(v);
                                        v.color    = vec3(0, 1, 0);
                                        v.position = center + planes[5].normal * 1.0;
                                        debugCluster.lines.push_back(v);
                    */
                }
            }
        }
    }

    if (renderDebugEnabled && debugFrustumToView)
    {
        debugCluster.lineWidth = 1;

        debugCluster.setModelMatrix(cam->getModelMatrix());  // is inverse view.
        debugCluster.translateLocal(vec3(0, 0, -0.0001f));
        debugPoints.setModelMatrix(cam->getModelMatrix());  // is inverse view.
#if 0
        debugCluster.setPosition(make_vec4(0));
        debugCluster.translateGlobal(vec3(0, 6, 0));
        debugCluster.setScale(make_vec3(0.33f));
#endif
        debugCluster.calculateModel();
        debugCluster.updateBuffer();
        debugFrustumToView = false;
        debugPoints.calculateModel();
        debugPoints.updateBuffer();
    }

    startTimer(0);
    itemBuffer.itemList.clear();
    int maxClusterItemsPerCluster = 256;  // TODO Paul: Hardcoded?
    itemBuffer.itemList.resize(maxClusterItemsPerCluster * culling_cluster.size());
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
    bool changed = false;
    changed |= ImGui::Checkbox("clusterThreeDimensional", &clusterThreeDimensional);
    changed |= ImGui::SliderInt("screenSpaceTileSize", &screenSpaceTileSize, 1, 1024);
    if (clusterThreeDimensional)
    {
        changed |= ImGui::SliderInt("depthSplits", &depthSplits, 1, 32);
    }
    else
        depthSplits = 1;


    clustersDirty |= changed | debugFrustumToView;

    ImGui::End();
}
}  // namespace Saiga
