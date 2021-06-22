/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/clusterer.h"

#include "saiga/core/imgui/imgui.h"


namespace Saiga
{
Clusterer::Clusterer(GLTimerSystem* timer) : timer(timer)
{
    clustersDirty = true;

    infoBuffer.create(infoBufferView, GL_DYNAMIC_DRAW);
    infoBuffer.bind(LIGHT_CLUSTER_INFO_BINDING_POINT);

    cached_projection = mat4::Identity();
}

Clusterer::~Clusterer() {}

void Clusterer::resize(int _width, int _height)
{
    if (width == _width && height == _height)
    {
        return;
    }
    width         = _width;
    height        = _height;
    clustersDirty = true;
}

void Clusterer::clusterLights(Camera* cam, const ViewPort& viewPort)
{
    clusterLightsInternal(cam, viewPort);
}

/*
void Clusterer::clusterLights(Camera* cam, const ViewPort& viewPort, ArrayView<PointLight*> pls, ArrayView<SpotLight*> sls)
{
    pointLightsClusterData.clear();
    spotLightsClusterData.clear();
    // Point Lights
    for (auto& pl : pls)
    {
        if (!pl->shouldRender()) continue;

        pointLightsClusterData.emplace_back(pl->position, pl->radius);
    }

    // Spot Lights
    for (auto& sl : sls)
    {
        if (!sl->shouldRender()) continue;

        float rad = radians(sl->getAngle());
        float l   = sl->radius;
        float radius;
        if (rad > pi<float>() * 0.25f)
            radius = l * tan(rad);
        else
            radius = l * 0.5f / (cos(rad) * cos(rad));
        vec3 world_center = sl->position + sl->direction.normalized() * radius;
        spotLightsClusterData.emplace_back(world_center, radius);
    }

    clusterLightsInternal(cam, viewPort);
}
*/

void Clusterer::renderDebug(Camera* cam)
{
    if (!clusterDebug) return;
    debugCluster.render(cam);
}

void Clusterer::imgui()
{
    if (ImGui::Begin("Clusterer"))
    {
        bool changed = false;
        ImGui::Text("resolution: %dx%d", width, height);

        changed |= ImGui::Checkbox("clusterThreeDimensional", &clusterThreeDimensional);
        changed |= ImGui::SliderInt("screenSpaceTileSize", &screenSpaceTileSize, 16, 1024);
        screenSpaceTileSize = std::max(screenSpaceTileSize, 16);
        if (clusterThreeDimensional)
        {
            changed |= ImGui::SliderInt("depthSplits", &depthSplits, 0, 127);
        }
        else
            depthSplits = 0;


        changed |= ImGui::Checkbox("useSpecialNearCluster", &useSpecialNearCluster);
        if (useSpecialNearCluster)
        {
            changed |= ImGui::SliderFloat("specialNearDepthPercent", &specialNearDepthPercent, 0.0f, 0.5f, "%.4f");
        }


        if (ImGui::Checkbox("clusterDebug", &clusterDebug) && clusterDebug) updateDebug = true;
        if (clusterDebug)
            if (ImGui::Button("updateDebug")) updateDebug = true;

        changed |= updateDebug;  // When debug is enabled the clusters are rebuild.

        changed |= ImGui::Checkbox("screenSpaceDebug", &screenSpaceDebug);
        changed |= ImGui::Checkbox("splitDebug", &splitDebug);

        static double sum = 0.0;
        if (timerIndex == 0)
        {
            sum = 0.0;
            for (int i = 0; i < 100; ++i)
            {
                sum += cpuAssignmentTimes[i] * 0.01;
            }
        }



        ImGui::Text("  %f ms %s", lightAssignmentTimer.getTimeMS(), "CPU Light Assignment");
        ImGui::Text("  %f ms %s", sum, "CPU Light Assignment (100 Avg.)");

        clustersDirty = changed;
    }
    ImGui::End();
}



}  // namespace Saiga
