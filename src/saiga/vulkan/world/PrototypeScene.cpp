/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/PrototypeScene.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/scene/GrassPass.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <chrono>
#include <iostream>

namespace Saiga
{
namespace Vulkan
{
void PrototypeScene::attach(RenderScene& scene, const std::vector<ForestRenderer::TreeInstance>& placed, uint32_t seed)
{
    detach();
    const auto t0 = std::chrono::steady_clock::now();
    renderScene   = &scene;
    settings.clamp();
    scene.setPrototypeEnvironment(settings.prototype(), seed);
    trees.init(scene, placed, seed, settings);
    std::cout << "PrototypeScene: " << trees.treeCount() << " trees in " << trees.setCount() << " sets in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
}

void PrototypeScene::detach()
{
    if (renderScene) trees.clear();
    trees       = PrototypeForest();
    renderScene = nullptr;
}

void PrototypeScene::update()
{
    if (!renderScene) return;
    settings.clamp();
    renderScene->setPrototypeSettings(settings.prototype());
    trees.apply(settings);
}

void PrototypeScene::imgui()
{
    if (!ImGui::Begin("World"))
    {
        ImGui::End();
        return;
    }
    // The frame and terrain statistics are in the app's "Engine statistics" (F1)
    ImGui::TextDisabled("frame times, terrain: Engine statistics (F1)");
    const auto ts = trees.stats();
    ImGui::Text("trees: %d in %d sets, drawn per LOD %u / %u / %u", trees.treeCount(), trees.setCount(), ts.visible[0],
                ts.visible[1], ts.visible[2]);
    if (renderScene)
    {
        const auto& gs = renderScene->grassPass().stats();
        ImGui::Text("grass: %d blocks, %lld clumps", gs.blocks, (long long)gs.clumps);
    }

    auto& st = settings;
    if (ImGui::CollapsingHeader("Passes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("terrain", &st.terrain);
        ImGui::SameLine();
        ImGui::Checkbox("trees", &st.trees);
        ImGui::SameLine();
        ImGui::Checkbox("grass", &st.grass);
        ImGui::Checkbox("water", &st.water);
        ImGui::SameLine();
        ImGui::Checkbox("sky", &st.sky);
        ImGui::SameLine();
        ImGui::Checkbox("fog", &st.fog);
        ImGui::Checkbox("flat shading", &st.flatShading);
        ImGui::SameLine();
        ImGui::Checkbox("depth pre-pass", &st.depthPrePass);
        ImGui::Combo("debug", &st.debugMode, "none\0lod tint\0\0");
    }
    if (ImGui::CollapsingHeader("Level of detail"))
    {
        ImGui::SliderFloat("terrain lod range", &st.terrainLodRange, 100, 2000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("morph start", &st.morphStart, 0.3f, 0.95f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree lod1", &st.treeLod1, 10, 500, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree lod2", &st.treeLod2, 50, 3000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree max distance", &st.treeMaxDistance, 500, 50000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass radius", &st.grassRadius, 0, 200, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass cell size", &st.grassCellSize, 0.1f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass density", &st.grassDensity, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("alpha cutoff", &st.alphaCutoff, 0.05f, 0.95f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    }
    if (ImGui::CollapsingHeader("Lighting"))
    {
        ImGui::SliderFloat("fovy", &st.fovy, 20, 110, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun azimuth", &st.sunAzimuth, -180, 180, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun elevation", &st.sunElevation, -10, 90, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun intensity", &st.sunIntensity, 0, 2, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("ambient", &st.ambient, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("albedo", &st.albedo, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("wind", &st.wind, 0, 4, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("fog density", &st.fogDensity, 0, 0.002f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("fog falloff", &st.fogFalloff, 0.0001f, 0.01f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
    }
    ImGui::End();
}

}  // namespace Vulkan
}  // namespace Saiga
