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

    cached_projection = mat4::Identity();

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

    if (ImGui::Checkbox("renderDebugEnabled", &renderDebugEnabled) && renderDebugEnabled) debugFrustumToView = true;
    if (renderDebugEnabled)
        if (ImGui::Button("debugFrustumToView")) debugFrustumToView = true;


    changed |= debugFrustumToView;

    changed |= ImGui::Checkbox("tileDebugView", &tileDebugView);
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
    changed |= ImGui::Checkbox("clusterThreeDimensional", &clusterThreeDimensional);
    changed |= ImGui::SliderInt("screenSpaceTileSize", &screenSpaceTileSize, 16, 1024);
    screenSpaceTileSize = std::max(screenSpaceTileSize, 16);
    if (clusterThreeDimensional)
    {
        changed |= ImGui::SliderInt("depthSplits", &depthSplits, 1, 128);
    }
    else
        depthSplits = 1;

        return changed;
}

void Clusterer::endImGui()
{
    ImGui::End();
}

}  // namespace Saiga
