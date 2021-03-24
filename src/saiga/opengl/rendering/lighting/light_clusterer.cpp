/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/light_clusterer.h"

#include "saiga/core/imgui/imgui.h"


namespace Saiga
{
Clusterer::Clusterer(GLTimerSystem* timer, ClustererParameters _params) : timer(timer)
{
    clusterThreeDimensional = _params.clusterThreeDimensional;
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
    clustersDirty = true;
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

    changed |= ImGui::Checkbox("clusterThreeDimensional", &clusterThreeDimensional);
    changed |= ImGui::SliderInt("screenSpaceTileSize", &screenSpaceTileSize, 16, 1024);
    screenSpaceTileSize = std::max(screenSpaceTileSize, 16);
    if (clusterThreeDimensional)
    {
        changed |= ImGui::SliderInt("depthSplits", &depthSplits, 0, 127);
    }
    else
        depthSplits = 0;

    if (ImGui::Checkbox("clusterDebug", &clusterDebug) && clusterDebug) updateDebug = true;
    if (clusterDebug)
        if (ImGui::Button("updateDebug")) updateDebug = true;

    changed |= updateDebug;  // When debug is enabled the clusters are rebuild.

    changed |= ImGui::Checkbox("screenSpaceDebug", &screenSpaceDebug);

    return changed;
}

void Clusterer::endImGui()
{
    ImGui::End();
}

}  // namespace Saiga
