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
 * JSON), Ambient (scale, floor, SH9, the prototype albedo colours) and GPU and memory; later parts add sky and clouds,
 * cascades and local lights.
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
    void gpuAndMemory();
    // Requests the preview of an image in this frame and draws the last one (registered once per preview texture as an
    // ImGui texture), at most maxWidth wide
    void previewImage(Vulkan::LightingImage image, bool log, float maxWidth);
    // The Exposure view's legend (a turbo colour bar of centre +- range / 2 EV100) and its centre / range controls
    void exposureLegend();
    EngineApp& app;
    // Per preview image: its ImGui texture and the preview version it was registered for; the ImGui renderer they were
    // registered with (a new renderer, e.g. a recreated scene, registers them again; the old one's went with it)
    std::array<ImTextureID, size_t(Vulkan::LightingImage::Count)> previewTextures{};
    std::array<uint64_t, size_t(Vulkan::LightingImage::Count)> previewVersions{};
    const void* previewImgui = nullptr;
    int environmentMip       = 0;
};

}  // namespace Engine
}  // namespace Saiga
