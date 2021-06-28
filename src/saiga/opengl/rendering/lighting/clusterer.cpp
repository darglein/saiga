/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/clusterer.h"

#include "saiga/core/imgui/imgui.h"


namespace Saiga
{
Clusterer::Clusterer(GLTimerSystem* timer, const ClustererParameters& _params) : timer(timer), params(_params)
{
    clustersDirty = true;

    infoBuffer.create(clusterInfoBuffer, GL_DYNAMIC_DRAW);
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

void Clusterer::clusterLights(Camera* cam, const ViewPort& viewPort, ArrayView<PointLight*> pls, ArrayView<SpotLight*> sls)
{
    lightsClusterData.clear();

    // Point Lights
    for (auto& pl : pls)
    {
        lightsClusterData.emplace_back(pl->position, pl->radius);
    }
    pointLightCount = pls.size();

    // Spot Lights
    for (auto& sl : sls)
    {
        float rad = radians(sl->getAngle());
        float l   = sl->radius;
        float radius;
        if (rad > pi<float>() * 0.25f)
            radius = l * tan(rad);
        else
            radius = l * 0.5f / (cos(rad) * cos(rad));
        vec3 world_center = sl->position + sl->direction.normalized() * radius;
        lightsClusterData.emplace_back(world_center, radius);
    }

    clusterLightsInternal(cam, viewPort);
}

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

        changed |= ImGui::Checkbox("clusterThreeDimensional", &params.clusterThreeDimensional);
        changed |= ImGui::SliderInt("screenSpaceTileSize", &params.screenSpaceTileSize, 16, 1024);
        params.screenSpaceTileSize = std::max(params.screenSpaceTileSize, 16);
        if (params.clusterThreeDimensional)
        {
            changed |= ImGui::SliderInt("depthSplits", &params.depthSplits, 0, 127);
        }
        else
            params.depthSplits = 0;


        changed |= ImGui::Checkbox("useSpecialNearCluster", &params.useSpecialNearCluster);
        if (params.useSpecialNearCluster)
        {
            changed |= ImGui::SliderFloat("specialNearDepthPercent", &params.specialNearDepthPercent, 0.0f, 0.5f, "%.4f");
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
