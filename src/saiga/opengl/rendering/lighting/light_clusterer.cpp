/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/light_clusterer.h"

#include "saiga/core/imgui/imgui.h"


namespace Saiga
{
Clusterer::Clusterer(GLTimerSystem* timer) : timer(timer)
{
    clustersDirty           = true;

    infoBuffer.createGLBuffer(nullptr, sizeof(infoBuf_t), GL_DYNAMIC_DRAW);
    clusterListBuffer.createGLBuffer(nullptr, 0, GL_DYNAMIC_DRAW);
    itemListBuffer.createGLBuffer(nullptr, 0, GL_DYNAMIC_DRAW);

    cached_projection = mat4::Identity();

    loadComputeShaders();
}

Clusterer::~Clusterer() {}

void Clusterer::init(int _width, int _height)
{
    width         = _width;
    height        = _height;
    clustersDirty = true;
}

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

void Clusterer::loadComputeShaders() {}

void Clusterer::renderImGui(bool* p_open)
{
    beginImGui(p_open);
    clustersDirty |= fillImGui();
    endImGui();
}


void Clusterer::beginImGui(bool* p_open)
{
    ImGui::Begin("Clusterer", p_open);
}

bool Clusterer::fillImGui()
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
        static float percent = 0.075f;
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

    return changed;
}

void Clusterer::endImGui()
{
    ImGui::End();
}

}  // namespace Saiga
