/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include <array>
#include <cstdint>

namespace Saiga
{
namespace Engine
{
class EngineApp;

/**
 * The debug UI's Lighting tab (F5; docs/engine_design.md §17.1; S36), set by the app with DebugUi::setTabContent:
 * Time and weather (the time of day, day and rate, the weather of the view camera's space; writes through setWorldTime
 * / setWeather between steps, so they replicate and appear in the log; with --coop on the hidden host, read-only on a
 * pure client), Sun and moon (directions, illuminance, the shadow light, the sun override) and Exposure (the EV100 of
 * the frame and of the curve, the compensation, the effective exposure and the pre-exposure; S36a1b: the luminance of
 * the frame (the Luminance preview and its reduction, computed only while the section is open) and the exposure
 * curve). S36a1b adds the toolbar (tone map, debug view, overlays, force full updates, the pipeline variant, copy
 * JSON), Ambient (scale, floor, SH9, the prototype albedo colours) and GPU and memory; S36a1c sky and clouds; S36a2
 * Cascades (the parameters, a row per cascade, the cascades' depth previews: a click on one enlarges it) and the
 * toolbar's "freeze cascades", Far terrain shadows (the horizon map: R / G preview, progress, direction, anchor, rows
 * per frame, recompute now); S36b Local lights (counts, the light table: a row selects and highlights a light, "move
 * camera to light", an entity light's LightSource fields written through set_component; S36b steps 3-4 add the
 * cluster heat map and the shadow atlas preview with its cache statistics).
 */
class SAIGA_ENGINE_API LightingTab
{
   public:
    explicit LightingTab(EngineApp& app) : app(app) {}
    // Releases the preview's ImGui texture (if the scene's ImGui renderer still exists)
    ~LightingTab();
    LightingTab(const LightingTab&)            = delete;
    LightingTab& operator=(const LightingTab&) = delete;
    // Inside the debug UI's panel (an ImGui frame)
    void draw();

   private:
    void toolbar();
    void timeAndWeather();
    void sunAndMoon();
    void exposure();
    void ambient();
    void skyAndClouds();
    void cascadesSection();
    void horizonSection();
    void localLightsSection();
    void gpuAndMemory();
    // Requests the preview of an image in this frame and draws the last one (registered once per preview texture as an
    // ImGui texture), at most maxWidth wide
    void previewImage(Vulkan::LightingImage image, bool log, float maxWidth, int mip = 0, int layer = 0);
    // The Exposure view's legend (a turbo colour bar of centre +- range / 2 EV100) and its centre / range controls
    void exposureLegend();
    EngineApp& app;
    // Per preview image: its ImGui texture and the preview version it was registered for; the ImGui renderer they were
    // registered with (a new renderer, e.g. a recreated scene, registers them again; the old one's went with it)
    std::array<ImTextureID, size_t(Vulkan::LightingImage::Count)> previewTextures{};
    std::array<uint64_t, size_t(Vulkan::LightingImage::Count)> previewVersions{};
    const void* previewImgui = nullptr;
    int environmentMip       = 0;
    // The enlarged cascade preview (-1: none); the horizon preview's channel (0 R, 1 G)
    int enlargedCascade = -1;
    int horizonChannel  = 0;
    // S36b: the light table's sort column and direction
    int lightSortColumn     = 3;
    bool lightSortAscending = true;
    int clusterSlice        = -1;
    // Draws a preview and returns whether it was clicked, with the click's position in [0, 1]^2 of the image
    bool previewImageClicked(Vulkan::LightingImage image, bool log, float maxWidth, ImVec2& at, int mip = 0,
                             int layer = 0);
};

}  // namespace Engine
}  // namespace Saiga
