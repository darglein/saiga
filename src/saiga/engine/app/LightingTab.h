/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Saiga
{
namespace Engine
{
class EngineApp;

// The Lighting tab's preview selectors beyond mip and layer (PreviewMapping: the cluster slice, the atlas tile's light
// and face)
struct LightingPreviewSelection
{
    int slice      = -1;
    uint32_t light = 0;
    int face       = 0;
};

// The camera pose of the Lighting tab's "move camera to light": back from the light (range) along the view's horizontal
// direction, or (spotDirection non-zero) behind and above a spot's cone, looking at the light; applied again from its
// own result it returns the same pose
SAIGA_ENGINE_API std::pair<Vec3, Quat> lightViewPose(const Vec3& light, const Vec3& spotDirection, double range,
                                                     const Quat& view);

/**
 * The debug UI's Lighting tab (F5; docs/engine_design.md §17.1; S36), set by the app with DebugUi::setTabContent. At
 * the top, not scrolled with the sections (UI rating round 1): a summary (tags of the states that change what the frame
 * shows, each a button that undoes it: CLAY, SUN OVERRIDE, CASCADES FROZEN, VIEW; the time, the sun and the moon, the
 * shadow light; the EV100, the frame's luminance, the visible local lights, the lighting's GPU time against its budget)
 * and one row of controls (the debug view, the overlays in a popup, copy JSON). Then the sections, only Time and
 * weather open at first; their details are sub-sections (DebugUi::subSection, collapsed):
 * Time and weather (the time of day, day and rate; the weather of the view camera's space; writes through setWorldTime
 * / setWeather between steps, so they replicate and appear in the log; with --coop on the hidden host, read-only on a
 * pure client), Sun and moon (directions, illuminance, the shadow light, the sun override), Exposure (the tone map, the
 * mode, the EV100 of the frame and of the curve, the compensation, the effective exposure; the luminance histogram, the
 * exposure curve with the current elevation, the Luminance preview), Ambient (scale, floor, SH L0; SH 1-8, the
 * prototype albedo, the environment map), Sky and clouds (the cloud layer; the atmosphere, the previews), Cascades (the
 * state, a row per cascade, the cascades' depth previews: a click on one enlarges it; split and resolution, bias,
 * casters), Far terrain shadows (the horizon map: progress, direction, anchor, R / G preview, recompute now), Local
 * lights (counts, the light table: a row selects and highlights a light, "move camera to light", an entity light's
 * LightSource fields written through set_component; entity lights that are not rendered (off, invalid, no pose) are
 * greyed rows with their state and stay editable), Light clusters (the grid, the occupancy, the heat map), Shadow atlas
 * (its parameters, the cache statistics, the preview with the tile owners) and GPU and memory (the GPU sections against
 * the budget, the memory, force full updates). The preview mapping is a popup above the first preview on screen.
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
    void summary();
    void controls();
    void timeAndWeather();
    void sunAndMoon();
    void exposure();
    void ambient();
    void skyAndClouds();
    void cascadesSection();
    void horizonSection();
    void localLightsSection();
    void lightTable();
    void clustersSection();
    void atlasSection();
    void gpuAndMemory();
    // Requests the preview of an image in this frame and draws the last one (registered once per preview texture as an
    // ImGui texture), at most maxWidth wide
    using Selection = LightingPreviewSelection;
    void previewImage(Vulkan::LightingImage image, bool log, float maxWidth, int mip = 0, int layer = 0,
                      const Selection& selection = Selection());
    // The preview mapping's button and popup (drawn once per frame, above the first preview that uses it)
    void previewMappingControl();
    bool mappingDrawn = false;
    // The Exposure view's legend (a turbo colour bar of centre +- range / 2 EV100) and its centre / range controls
    void exposureLegend();
    // Every write of the settings and parameters goes through validateLighting (round 2, finding 9: the tab wrote
    // without set_lighting's checks, e.g. a typed slider value out of range or a cluster grid beyond the memory limit);
    // a refused write keeps the state and shows its error (writeError) until the next accepted one
    bool applySettings(const Vulkan::EnvironmentSettings& s);
    bool applyParameters(const Vulkan::LightingParameters& p);
    std::string writeError;
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
    int atlasTileFace       = 0;
    int atlasLayer          = 0;
    // The selected entity light (round 1, finding 8: kept by entity, so it stays selected while switched off; its sink
    // handle is the debug state's highlightLight while rendered) and the highlight the tab set last (another change of
    // highlightLight, e.g. an atlas tile click or MCP, selects that light's entity)
    Entity selectedEntity;
    uint32_t tabHighlight = 0;
    // EngineApp::lightHighlightClears seen by the tab
    uint64_t seenHighlightClears = 0;
    // The table's order (round 1, finding 17: sorted when the sort spec or the light set changes, else at most every
    // 30 frames for the distances): row keys (a handle, or 1 << 32 | entity index for an unrendered entity light)
    std::vector<uint64_t> lightOrder;
    int orderColumn     = -1;
    bool orderAscending = true;
    uint64_t orderFrame = 0;
    // Draws a preview and returns whether it was clicked, with the click's position in [0, 1]^2 of the image
    bool previewImageClicked(Vulkan::LightingImage image, bool log, float maxWidth, ImVec2& at, int mip = 0,
                             int layer = 0, const Selection& selection = Selection());
};

}  // namespace Engine
}  // namespace Saiga
