/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The Lighting tab (F5) of the debug UI.

#include "saiga/engine/app/LightingTab.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/app/DebugUi.h"
#include "saiga/engine/app/DebugUiStyle.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/app/LightingTools.h"
#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/scene/EnvironmentMap.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/LocalLights.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/ShadowAtlas.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
namespace
{
// The turbo colour map of the Exposure view (debug_common.glsl's polynomial)
ImU32 turboColor(float t)
{
    t              = std::clamp(t, 0.0f, 1.0f);
    const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
    auto c = [&](float a, float b, float d, float e, float f, float g)
    { return std::clamp(a + b * t + d * t2 + e * t3 + f * t4 + g * t5, 0.0f, 1.0f); };
    return ImGui::ColorConvertFloat4ToU32(
        ImVec4(c(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f, -152.94239396f, 59.28637943f),
               c(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f, 4.27729857f, 2.82956604f),
               c(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f, -89.90310912f, 27.34824973f), 1.0f));
}

// The debug views and the overlays under readable labels (UI rating round 1, findings 3 and 5: the MCP names, e.g.
// cluster_lights, were the labels; they are in the tooltips now) and what each shows. A state that also has a control
// in a section (light volumes, cluster grid, atlas tile owners, the cluster heat map) has the same label there
const char* const kViewLabels[] = {"none",           "lighting only",    "albedo",          "normals",
                                   "roughness",      "metalness",        "material",        "direct only",
                                   "ambient only",   "specular only",    "cloud shadow",    "shadow factor",
                                   "exposure",       "shadow cascades",  "horizon shadow",  "local lights only",
                                   "cluster heat map", "atlas shadow"};
const char* const kViewTooltips[] = {
    "The lit frame",
    "The lighting on a white surface (roughness 0.5)",
    "The surfaces' albedo (raw)",
    "The shading normals (n * 0.5 + 0.5)",
    "The roughness (raw)",
    "The metalness (raw)",
    "Roughness, metalness and ambient occlusion as red, green and blue",
    "The direct light of the sun and the moon alone",
    "The diffuse part of the ambient alone",
    "The specular reflection of the environment alone",
    "The cloud shadow factor (white: no shadow)",
    "The shadow factor of the shadow light (white: lit)",
    "The luminance as a heat map in EV100 (its legend below the controls)",
    "The cascade shadow tinted by its cascade (red, green, blue, yellow; gray: outside every cascade)",
    "The far terrain shadow (the horizon map's factor)",
    "The direct light of the local lights alone",
    "The lights of each fragment's cluster (turbo; magenta: an overflowed cluster)",
    "The atlas shadow of the local lights (the selected light's, else the darkest; blue: no shadowed light)"};
static_assert(sizeof(kViewLabels) / sizeof(kViewLabels[0]) == size_t(Vulkan::LightingDebugView::Count), "views");
static_assert(sizeof(kViewTooltips) / sizeof(kViewTooltips[0]) == size_t(Vulkan::LightingDebugView::Count), "views");
// In the order of lightingOverlayNames
const char* const kOverlayLabels[]   = {"sun direction",  "cascade colors",        "cascade bounds",
                                        "camera frustum splits", "horizon anchor", "light volumes",
                                        "cluster grid",   "atlas tile owners"};
const char* const kOverlayTooltips[] = {
    "Lines from the camera's focus towards the sun and the moon",
    "The surfaces tinted by their cascade (also with the other views)",
    "Each cascade's box and caster volume as a wireframe in its color",
    "The split planes of the camera frustum the cascades were fitted to",
    "The horizon map's extent and its inner half (the map moves when the camera leaves it)",
    "The volumes of the frame's lights (the selected light white)",
    "The depth slices and tiles of the light clusters",
    "The owner (light and face) of each tile in the shadow atlas preview"};
constexpr size_t kOverlays = sizeof(kOverlayLabels) / sizeof(kOverlayLabels[0]);

// An overlay's checkbox (the overlays popup and the sections draw the same one)
void overlayCheckbox(Vulkan::RenderScene& scene, uint32_t flag)
{
    const auto& names = Vulkan::lightingOverlayNames();
    for (size_t i = 0; i < names.size() && i < kOverlays; ++i)
    {
        if (names[i].second != flag) continue;
        Vulkan::LightingDebugState d = scene.lightingDebug();
        bool on                      = (d.overlays & flag) != 0;
        if (ImGui::Checkbox(kOverlayLabels[i], &on))
        {
            d.overlays = on ? (d.overlays | flag) : (d.overlays & ~flag);
            scene.setLightingDebug(d);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n(overlay %s)", kOverlayTooltips[i], names[i].first);
    }
}

// Continues the line with an item w wide if it fits the region, else the item starts a new line (and always for the
// first item of a line)
void flowItem(float w, bool first)
{
    if (first) return;
    const float spacing = 0.6f * ImGui::GetFontSize();
    const float lineEnd = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    if (ImGui::GetItemRectMax().x + spacing + w <= lineEnd) ImGui::SameLine(0, spacing);
}

void flowText(ImU32 color, const char* text, bool first = false)
{
    flowItem(ImGui::CalcTextSize(text).x, first);
    DebugUiStyle::coloredText(color, text);
}

// A state tag of the summary (a state that changes what the frame shows and is easily forgotten): a small button whose
// click undoes the state
bool tagButton(const char* label, const char* tooltip, bool first)
{
    flowItem(ImGui::CalcTextSize(label).x + 2 * ImGui::GetStyle().FramePadding.x, first);
    ImGui::PushStyleColor(ImGuiCol_Text, DebugUiStyle::kWarning);
    const bool clicked = ImGui::SmallButton(label);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
    return clicked;
}

// Right-aligned formatted text in the next table cell
void cellRightf(const char* format, ...) IM_FMTARGS(1);
void cellRightf(const char* format, ...)
{
    char text[64];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    DebugUiStyle::cellRight(text);
}

const char* dayPhase(double sunElevationDeg, double timeOfDay)
{
    if (sunElevationDeg >= 6) return "day";
    if (sunElevationDeg >= 0) return "low sun";
    if (sunElevationDeg >= -18) return timeOfDay < 12 ? "dawn" : "dusk";
    return "night";
}

constexpr double kDegrees = 57.29577951308232;
// The selected light (its table row, its tile frame in the atlas preview)
constexpr ImU32 kSelectedLight = IM_COL32(255, 220, 60, 255);
}  // namespace

std::pair<Vec3, Quat> lightViewPose(const Vec3& light, const Vec3& spotDirection, double range, const Quat& view)
{
    // Back from the light along the view's horizontal direction (a spot: behind and above its cone), looking at it.
    // Regression (S36 demo): the view's full forward was used, so every further click of "move camera to light" (the
    // camera then looking down at the light) moved the eye higher
    Vec3 forward = view * Vec3(0, 0, -1);
    forward(1)   = 0;
    if (forward.norm() < 1e-6) forward = Vec3(0, 0, -1);
    forward.normalize();
    if (spotDirection.norm() > 1e-9) forward = spotDirection.normalized();
    const double back = std::max(2.0, 0.6 * range);
    const Vec3 eye    = light - forward * back + Vec3(0, 0.3 * back, 0);
    return {eye, cameraLookRotation(light - eye)};
}

LightingTab::~LightingTab()
{
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || scene->imguiRenderer() != previewImgui) return;
    for (ImTextureID t : previewTextures)
        if (t) scene->imguiRenderer()->removeTexture(t);
}

void LightingTab::draw()
{
    // The summary and the controls stay at the top; the sections scroll below them (UI rating round 1, finding 15)
    mappingDrawn = false;
    summary();
    controls();
    ImGui::Spacing();
    if (ImGui::BeginChild("##lighting body", ImVec2(0, 0), ImGuiChildFlags_None))
    {
        timeAndWeather();
        sunAndMoon();
        exposure();
        ambient();
        skyAndClouds();
        cascadesSection();
        horizonSection();
        localLightsSection();
        clustersSection();
        atlasSection();
        gpuAndMemory();
    }
    ImGui::EndChild();
}

void LightingTab::previewImage(Vulkan::LightingImage image, bool log, float maxWidth, int mip, int layer,
                               const Selection& selection)
{
    ImVec2 at;
    previewImageClicked(image, log, maxWidth, at, mip, layer, selection);
}

bool LightingTab::previewImageClicked(Vulkan::LightingImage image, bool log, float maxWidth, ImVec2& at, int mip,
                                      int layer, const Selection& selection)
{
    // The mapping's control above the frame's first preview that uses it (the clusters preview has its own)
    if (!mappingDrawn && image != Vulkan::LightingImage::Clusters) previewMappingControl();
    Vulkan::RenderScene* scene     = app.scene();
    Vulkan::PreviewMapping mapping = app.lightingPreviewMapping();
    mapping.log                    = log;  // the luminance passes the set_lighting_debug mapping's log
    mapping.mip                    = mip;
    mapping.layer                  = layer;
    mapping.slice                  = selection.slice;
    mapping.light                  = selection.light;
    mapping.face                   = selection.face;
    // Converted only while it is on screen (not scrolled out of the panel), at preview resolution and a few times per
    // second (LightingDebug::show)
    Vulkan::Texture2D* preview = scene->lightingDebugImages().preview(image);
    if (preview)
    {
        const float w = std::min(ImGui::GetContentRegionAvail().x, maxWidth);
        const ImVec2 size(w, w * float(preview->height) / float(std::max(preview->width, 1u)));
        if (!ImGui::IsRectVisible(size))
        {
            ImGui::Dummy(size);
            return false;
        }
    }
    scene->lightingDebugImages().show(image, mapping);
    Vulkan::ImGuiVulkanRenderer* imgui = scene->imguiRenderer();
    if (!imgui || !preview) return false;
    // A new ImGui renderer (a recreated scene): the old one's textures went with it
    if (imgui != previewImgui)
    {
        previewTextures.fill(ImTextureID(0));
        previewVersions.fill(0);
        previewImgui = imgui;
    }
    const size_t k         = size_t(image);
    const uint64_t version = scene->lightingDebugImages().previewVersion(image);
    if (version != previewVersions[k] || !previewTextures[k])
    {
        if (previewTextures[k]) imgui->removeTexture(previewTextures[k]);
        previewTextures[k] = imgui->addTexture(preview->view(), preview->memoryLocation->data.sampler,
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
        previewVersions[k] = version;
    }
    const float w = std::min(ImGui::GetContentRegionAvail().x, maxWidth);
    const ImVec2 size(w, w * float(preview->height) / float(std::max(preview->width, 1u)));
    ImGui::Image(previewTextures[k], size);
    if (!ImGui::IsItemClicked()) return false;
    const ImVec2 lo = ImGui::GetItemRectMin(), mouse = ImGui::GetMousePos();
    at = ImVec2((mouse.x - lo.x) / std::max(size.x, 1.0f), (mouse.y - lo.y) / std::max(size.y, 1.0f));
    return true;
}

void LightingTab::previewMappingControl()
{
    // The previews' mapping (set_lighting_debug mapping) in a popup: it matters only next to a preview (UI rating round
    // 1, finding 3: it was in the toolbar, always visible)
    mappingDrawn              = true;
    Vulkan::PreviewMapping& m = app.lightingPreviewMapping();
    char label[96];
    if (m.autoRange)
        std::snprintf(label, sizeof(label), "preview mapping: auto range, %s",
                      m.colorMap == Vulkan::ColorMap::Turbo ? "turbo" : "gray");
    else
        std::snprintf(label, sizeof(label), "preview mapping: %.3g .. %.3g, %s", double(m.min), double(m.max),
                      m.colorMap == Vulkan::ColorMap::Turbo ? "turbo" : "gray");
    // At the right of the preview's label line if it fits, else on a line of its own
    const float w       = ImGui::CalcTextSize(label).x + 2 * ImGui::GetStyle().FramePadding.x;
    const float lineEnd = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    if (ImGui::GetItemRectMax().x + ImGui::GetFontSize() + w <= lineEnd)
    {
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(lineEnd - w, ImGui::GetCursorScreenPos().y));
    }
    if (ImGui::SmallButton(label)) ImGui::OpenPopup("##preview mapping");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("How the previews map their values to colors (click to change)");
    if (ImGui::BeginPopup("##preview mapping"))
    {
        ImGui::Checkbox("auto range", &m.autoRange);
        if (!m.autoRange)
        {
            float range[2] = {m.min, m.max};
            ImGui::SetNextItemWidth(140);
            // (a finite span: the shader maps and bins over max - min; CodeRabbit #86 round 2)
            if (ImGui::InputFloat2("##preview range", range, "%.3g") && std::isfinite(range[0]) &&
                std::isfinite(range[1]) && range[0] < range[1] && std::isfinite(range[1] - range[0]))
            {
                m.min = range[0];
                m.max = range[1];
            }
        }
        ImGui::Checkbox("log2 (the luminance)", &m.log);
        bool turbo = m.colorMap == Vulkan::ColorMap::Turbo;
        if (ImGui::Checkbox("turbo colors", &turbo))
            m.colorMap = turbo ? Vulkan::ColorMap::Turbo : Vulkan::ColorMap::Gray;
        ImGui::EndPopup();
    }
}

void LightingTab::summary()
{
    // What the frame is doing at a glance, before any section (UI rating round 1, finding 1): the state tags, the time,
    // the sun and the moon, the shadow light; the exposure, the frame's luminance, the local lights, the lighting's GPU
    // time against its budget
    Vulkan::RenderScene* scene = app.scene();
    if (!scene) return;
    DebugUi& ui                          = app.debugUi();
    const EnvironmentSample e            = app.sampleViewEnvironment();
    const CelestialState& c              = e.celestial;
    Vulkan::EnvironmentSettings s        = app.environmentSettings();
    Vulkan::LightingDebugState d         = scene->lightingDebug();
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    const Vulkan::LightingParameters& p  = scene->lightingParameters();
    char text[128];
    bool first = true;
    if (scene->parameters().toneMap == Vulkan::ToneMap::Identity)
    {
        if (tagButton("CLAY", "Clay: the studio light, the time of day has no effect.\nClick: the ACES tone map",
                      first))
            scene->setToneMap(Vulkan::ToneMap::Aces);
        ui.markItem("lighting/tag clay");
        first = false;
    }
    if (s.sunOverride)
    {
        if (tagButton("SUN OVERRIDE", "The sun's direction is overridden (Sun and moon).\nClick: the sun of the time",
                      first))
        {
            s.sunOverride.reset();
            applySettings(s);
        }
        ui.markItem("lighting/tag sun override");
        first = false;
    }
    if (scene->cascadesFrozen())
    {
        if (tagButton("CASCADES FROZEN", "The cascades keep their frozen camera (Cascades).\nClick: unfreeze", first))
            scene->setFreezeCascades(false);
        ui.markItem("lighting/tag frozen");
        first = false;
    }
    if (d.view != Vulkan::LightingDebugView::None && d.view < Vulkan::LightingDebugView::Count)
    {
        std::snprintf(text, sizeof(text), "VIEW: %s", kViewLabels[size_t(d.view)]);
        if (tagButton(text, "A debug view replaces the frame.\nClick: the lit frame (view none)", first))
        {
            d.view = Vulkan::LightingDebugView::None;
            scene->setLightingDebug(d);
        }
        ui.markItem("lighting/tag view");
        first = false;
    }

    // The time, the sun and the moon, the shadow light
    const int minutes = int(std::floor(e.time.timeOfDay * 60.0));
    std::snprintf(text, sizeof(text), "%02d:%02d day %lld", (minutes / 60) % 24, minutes % 60, (long long)e.time.day);
    flowText(DebugUiStyle::kWhite, e.hasTime ? text : "no world time (noon)", first);
    if (e.hasTime && e.time.rate != 1)
    {
        if (e.time.rate == 0)
            std::snprintf(text, sizeof(text), "(stopped)");
        else
            std::snprintf(text, sizeof(text), "x%g", e.time.rate);
        flowText(DebugUiStyle::kRef, text);
    }
    const SpaceId space = app.viewCamera().space;
    if (e.enclosed)
    {
        std::snprintf(text, sizeof(text), "space %d (enclosed: no sun or sky)", int(space));
        flowText(DebugUiStyle::kRef, text);
    }
    else
    {
        flowText(DebugUiStyle::kRef, dayPhase(c.sunElevationDeg, e.time.timeOfDay));
        std::snprintf(text, sizeof(text), "sun %.1f\xc2\xb0", c.sunElevationDeg);
        flowText(DebugUiStyle::kWhite, text);
        std::snprintf(text, sizeof(text), "moon %.1f\xc2\xb0", c.moonElevationDeg);
        flowText(DebugUiStyle::kWhite, text);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("phase %.2f", c.moonPhase);
        std::snprintf(text, sizeof(text), "shadow: %s", shadowLightName(e.shadowLight));
        flowText(DebugUiStyle::kWhite, text);
        if (space != 0)
        {
            std::snprintf(text, sizeof(text), "space %d", int(space));
            flowText(DebugUiStyle::kRef, text);
        }
    }

    // The exposure, the luminance (its statistics are requested here: the Luminance preview's reduction at preview
    // resolution, a few times per second while the tab is shown), the local lights, the GPU time
    first = true;
    if (scene->hasEnvironment())
    {
        std::snprintf(text, sizeof(text), "EV100 %.2f", double(scene->environment().ev100));
        flowText(DebugUiStyle::kWhite, text, true);
        first = false;
        if (s.exposure.mode == Vulkan::ExposureMode::Fixed)
            std::snprintf(text, sizeof(text), "(fixed)");
        else if (s.exposure.compensation != 0)
            std::snprintf(text, sizeof(text), "(curve %+.1f EV)", double(s.exposure.compensation));
        else
            std::snprintf(text, sizeof(text), "(curve)");
        flowText(DebugUiStyle::kRef, text);
        // (the request of the Exposure section's preview: one conversion per frame)
        Vulkan::PreviewMapping luminance = app.lightingPreviewMapping();
        luminance.mip = luminance.layer = luminance.face = 0;
        luminance.slice                                    = -1;
        luminance.light                                    = 0;
        scene->lightingDebugImages().show(Vulkan::LightingImage::Luminance, luminance);
        if (ls.luminanceFrame > 0 && ls.luminanceFrame + 600 >= scene->frames())
        {
            const bool log = app.lightingPreviewMapping().log;
            std::snprintf(text, sizeof(text), log ? "lum %.1f / %.1f log2" : "lum %.3g / %.3g",
                          double(ls.luminance.mean), double(ls.luminance.max));
            flowText(DebugUiStyle::kWhite, text);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("The frame's mean / max luminance (%s; Exposure has the histogram)",
                                  scene->sceneWritesLinear() ? "cd/m^2" : "display values");
        }
    }
    if (p.localLights)
    {
        const Vulkan::LocalLightStatistics& l = ls.lights;
        std::snprintf(text, sizeof(text), "lights %u/%u", l.visible, l.submitted);
        const bool warn = l.overBudget > 0 || l.overflowedClusters > 0;
        flowText(warn ? DebugUiStyle::kWarn : DebugUiStyle::kWhite, text, first);
        first = false;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Visible / submitted local lights; %u over budget, %u overflowed clusters", l.overBudget,
                              l.overflowedClusters);
    }
    const LightingGpuTimes t = lightingGpuTimes(app);
    std::snprintf(text, sizeof(text), "lighting %.2f / %.1f ms", t.lightingMs, LightingGpuTimes::kBudgetMs);
    flowText(t.lightingMs > LightingGpuTimes::kBudgetMs ? DebugUiStyle::kWarn : DebugUiStyle::kGpu, text, first);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The lighting's GPU time (mean of %d frames) against its budget (GPU and memory)",
                          LightingGpuTimes::kMeanFrames);
    lightingGraph(t);
}

void LightingTab::lightingGraph(const LightingGpuTimes& t)
{
    // The lighting's GPU time of every kept frame (the user's request after the S36 demo): a line in the GPU color, the
    // budget as a dashed line (red frames above it), the scale from 0 to the larger of 1.25 x the budget and the
    // highest frame; hovering shows a frame's time
    const std::vector<double>& h = t.lightingHistory;
    if (h.size() < 2) return;
    const float em     = ImGui::GetFontSize();
    const float width  = ImGui::GetContentRegionAvail().x;
    const float height = 3 * em;
    if (width < 4 * em) return;
    double top = 1.25 * LightingGpuTimes::kBudgetMs;
    for (double v : h) top = std::max(top, v);
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##lighting graph", ImVec2(width, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec2 p1(p0.x + width, p0.y + height);
    dl->AddRectFilled(p0, p1, IM_COL32(255, 255, 255, 12));
    auto y = [&](double ms) { return p1.y - float(ms / top) * (height - 1); };
    const float budgetY = y(LightingGpuTimes::kBudgetMs);
    for (float x = p0.x; x < p1.x; x += 8)
        dl->AddLine(ImVec2(x, budgetY), ImVec2(std::min(x + 4, p1.x), budgetY), DebugUiStyle::kWarn & 0xA0FFFFFF);
    const float step = width / float(h.size() - 1);
    for (size_t i = 1; i < h.size(); ++i)
    {
        const ImVec2 a(p0.x + step * float(i - 1), y(h[i - 1]));
        const ImVec2 b(p0.x + step * float(i), y(h[i]));
        const bool over = h[i] > LightingGpuTimes::kBudgetMs || h[i - 1] > LightingGpuTimes::kBudgetMs;
        dl->AddLine(a, b, over ? DebugUiStyle::kWarn : DebugUiStyle::kGpu, 1.5f);
    }
    char text[48];
    std::snprintf(text, sizeof(text), "%.1f ms", top);
    dl->AddText(ImVec2(p0.x + 3, p0.y + 1), IM_COL32(255, 255, 255, 120), text);
    if (ImGui::IsItemHovered())
    {
        const float mx = ImGui::GetIO().MousePos.x - p0.x;
        const size_t i = std::min(h.size() - 1, size_t(std::max(0.0f, mx / step + 0.5f)));
        const float x  = p0.x + step * float(i);
        dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y), IM_COL32(255, 255, 255, 90));
        ImGui::SetTooltip("lighting GPU time: %.3f ms (frame %d of the last %d; budget %.1f ms)", h[i],
                          int(i) - int(h.size()) + 1, int(h.size()), LightingGpuTimes::kBudgetMs);
    }
}

void LightingTab::controls()
{
    // One row (UI rating round 1, finding 3): the debug view, the overlays in a popup, copy JSON at the right. The tone
    // map is in Exposure, force full updates in GPU and memory, freeze in Cascades, the preview mapping next to the
    // previews
    Vulkan::RenderScene* scene = app.scene();
    if (!scene) return;
    Vulkan::LightingDebugState d = scene->lightingDebug();
    const float em               = ImGui::GetFontSize();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("view");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(8 * em);
    const size_t view = std::min(size_t(d.view), size_t(Vulkan::LightingDebugView::Count) - 1);
    if (ImGui::BeginCombo("##debug view", kViewLabels[view]))
    {
        for (size_t i = 0; i < size_t(Vulkan::LightingDebugView::Count); ++i)
        {
            if (ImGui::Selectable(kViewLabels[i], i == view))
            {
                d.view = Vulkan::LightingDebugView(i);
                scene->setLightingDebug(d);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s\n(view %s)", kViewTooltips[i],
                                  Vulkan::lightingDebugViewName(Vulkan::LightingDebugView(i)));
        }
        ImGui::EndCombo();
    }
    else if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The debug view: %s", kViewTooltips[view]);
    ImGui::SameLine();
    int count = 0;
    for (uint32_t bits = d.overlays & Vulkan::OverlayAll; bits; bits &= bits - 1) ++count;
    char label[32];
    std::snprintf(label, sizeof(label), "overlays (%d)", count);
    ImGui::SetNextItemWidth(7 * em);
    if (ImGui::BeginCombo("##overlays", label))
    {
        for (const auto& o : Vulkan::lightingOverlayNames()) overlayCheckbox(*scene, o.second);
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    const float w = ImGui::CalcTextSize("copy JSON").x + 2 * ImGui::GetStyle().FramePadding.x;
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - w));
    if (ImGui::Button("copy JSON"))
    {
        json11::Json out;
        std::string error;
        if (lightingJson(app, {}, out, error)) ImGui::SetClipboardText(out.dump().c_str());
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copies get_lighting's answer to the clipboard");
    // The last refused write (validateLighting: a typed value out of range, a cross-key rule, the memory)
    if (!writeError.empty())
    {
        ImGui::PushTextWrapPos(0);
        DebugUiStyle::coloredText(DebugUiStyle::kError, ("not applied: " + writeError).c_str());
        ImGui::PopTextWrapPos();
    }
    if (d.view == Vulkan::LightingDebugView::Exposure) exposureLegend();
}

bool LightingTab::applySettings(const Vulkan::EnvironmentSettings& s)
{
    Vulkan::RenderScene* scene = app.scene();
    std::string error;
    // The settings' values only: the parameters are unchanged, so their memory and device limits are not checked again
    // (CodeRabbit #86 round 2: an environment file's cluster slots, reduced by the scene to the device's limit, refused
    // every exposure edit)
    const bool ok =
        validateLightingValues(s, scene ? scene->lightingParameters() : Vulkan::LightingParameters(), error);
    if (!ok)
    {
        writeError = error;
        return false;
    }
    writeError.clear();
    app.setEnvironmentSettings(s);
    return true;
}

bool LightingTab::applyParameters(const Vulkan::LightingParameters& p)
{
    Vulkan::RenderScene* scene = app.scene();
    if (!scene) return false;
    std::string error;
    if (!validateLighting(scene->lightingStatistics(), scene->storageBufferLimit(), app.environmentSettings(), p,
                          error))
    {
        writeError = error;
        return false;
    }
    writeError.clear();
    scene->setLightingParameters(p);
    return true;
}

void LightingTab::exposureLegend()
{
    Vulkan::RenderScene* scene        = app.scene();
    const std::optional<float> centre = scene->exposureViewCentreSetting();
    const float range                 = scene->exposureViewRangeSetting();
    const float c                     = scene->frameExposureViewCentre();
    // The colour bar: the heat map's turbo over [c - range / 2, c + range / 2] EV100
    const ImVec2 p0    = ImGui::GetCursorScreenPos();
    const float width  = std::max(ImGui::GetContentRegionAvail().x, 64.0f);
    const float height = 12;
    ImDrawList* list   = ImGui::GetWindowDrawList();
    const int n        = 32;
    for (int i = 0; i < n; ++i)
    {
        const float x0 = p0.x + width * float(i) / n, x1 = p0.x + width * float(i + 1) / n;
        list->AddRectFilled(ImVec2(x0, p0.y), ImVec2(x1, p0.y + height), turboColor((float(i) + 0.5f) / n));
    }
    ImGui::Dummy(ImVec2(width, height));
    ImGui::Text("%.1f EV", double(c - 0.5f * range));
    ImGui::SameLine(std::max(width * 0.5f - 30, 0.0f));
    ImGui::Text("%.1f EV100%s", double(c), centre ? "" : " (auto)");
    ImGui::SameLine(std::max(width - 50, 0.0f));
    ImGui::Text("%.1f EV", double(c + 0.5f * range));
    // The centre (auto: the frame's EV100, the display mid-gray in clay) and the range
    bool automatic = !centre;
    bool changed   = ImGui::Checkbox("auto centre", &automatic);
    float value    = centre ? *centre : c;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    if (automatic) ImGui::BeginDisabled();
    changed = ImGui::SliderFloat("##exposure view centre", &value, -10, 20, "centre %.1f") || changed;
    if (automatic) ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    float r = range;
    changed = ImGui::SliderFloat("##exposure view range", &r, 0.5f, 64, "range %.1f EV") || changed;
    if (changed) scene->setExposureViewRange(automatic ? std::nullopt : std::optional<float>(value), r);
}

void LightingTab::timeAndWeather()
{
    DebugUi& ui = app.debugUi();
    if (!ui.section("lighting/Time and weather", "Time and weather", true)) return;
    World& target = lightingWriteWorld(app);
    if (!hasWorldTime(target))
    {
        DebugUiStyle::coloredText(DebugUiStyle::kRef, "the world has no WorldTime (EnvironmentSetup::enabled)");
        return;
    }
    const SpaceId space = app.viewCamera().space;
    // The edit widgets show the values of the world they write (with --coop the hidden host, at its latest step): the
    // client's sample lags by the replication delay, and an incremental widget (drag, +/-) would add its delta to the
    // stale value and jump back and forth until the replication catches up
    const EnvironmentSample e =
        sampleEnvironment(target, space, 1.0, app.environmentSettings(), target.environmentSetup());
    // A pure client cannot write the world state (the host does, R-NET-2)
    const bool writable = target.isHost();
    EngineApp* a        = &app;
    // A refused write (e.g. a range) disappears from the widget: log it like the tools do
    auto queueTime = [a](WorldTimeChange c)
    {
        a->queueBetweenSteps(
            [a, c]()
            {
                std::string error;
                if (!setWorldTime(lightingWriteWorld(*a), c, error))
                    SAIGA_LOG("api", "error lighting tab set_time_of_day: %s", error.c_str());
            });
    };
    auto queueWeather = [a, space](WeatherChange c)
    {
        a->queueBetweenSteps(
            [a, space, c]()
            {
                std::string error;
                if (!setWeather(lightingWriteWorld(*a), space, c, error))
                    SAIGA_LOG("api", "error lighting tab set_weather: %s", error.c_str());
            });
    };
    if (!writable) ImGui::BeginDisabled();
    if (DebugUiStyle::beginProperties("##lighting time"))
    {
        DebugUiStyle::property("time of day");
        float hours = float(e.time.timeOfDay);
        if (ImGui::SliderFloat("##time of day", &hours, 0.0f, 23.99f, "%.2f h"))
            queueTime(WorldTimeChange{double(hours), std::nullopt, std::nullopt});
        DebugUiStyle::property("day");
        // The full int64 day (set_time_of_day accepts +-1e15)
        int64_t day       = e.time.day;
        const int64_t one = 1;
        if (ImGui::InputScalar("##day", ImGuiDataType_S64, &day, &one))
            queueTime(WorldTimeChange{std::nullopt, day, std::nullopt});
        // The rate's buttons, the current one highlighted (UI rating round 1, finding 7: the rate was printed as "x0"
        // before the buttons x0 x1 ..., which read as one more button)
        DebugUiStyle::property("rate");
        bool listed = false;
        for (double r : {0.0, 1.0, 60.0, 600.0})
        {
            if (r != 0.0) ImGui::SameLine();
            const std::string label = "x" + std::to_string(int(r));
            const bool current      = r == e.time.rate;
            listed                  = listed || current;
            if (current) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
            const bool clicked = ImGui::SmallButton(label.c_str());
            if (current) ImGui::PopStyleColor();
            ui.markItem("lighting/rate " + label);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%.0f game seconds per real second%s", r, r == 0 ? " (the time stands still)" : "");
            if (clicked) queueTime(WorldTimeChange{std::nullopt, std::nullopt, r});
        }
        if (!listed)
        {
            ImGui::SameLine();
            ImGui::TextDisabled("now x%g", e.time.rate);
        }
        if (&target != &app.world())
        {
            // --coop: the rendered client's time (it follows the host by the replication delay)
            const WorldTime view = app.sampleViewEnvironment().time;
            DebugUiStyle::property("client view");
            ImGui::Text("%.2f h, day %lld", view.timeOfDay, (long long)view.day);
        }
        DebugUiStyle::endProperties();
    }
    if (!writable) ImGui::EndDisabled();
    // The weather in a sub-section (UI rating round 1, finding 2: the section was 15 rows; the time is what is changed
    // most often)
    if (hasWeather(target) && ui.subSection("lighting/Time and weather: Weather", "Weather", false))
    {
        if (!writable) ImGui::BeginDisabled();
        if (DebugUiStyle::beginProperties("##lighting weather"))
        {
            const Weather& w = e.weather;
            auto slider      = [&](const char* label, double value, float lo, float hi,
                              std::optional<double> WeatherChange::* f, ImGuiSliderFlags flags = 0,
                              const char* format = "%.3f")
            {
                DebugUiStyle::property(label);
                float v = float(value);
                if (ImGui::SliderFloat((std::string("##") + label).c_str(), &v, lo, hi, format, flags))
                {
                    WeatherChange c;
                    c.*f = double(v);
                    queueWeather(c);
                }
            };
            slider("cloudiness", w.cloudiness, 0, 1, &WeatherChange::cloudiness);
            slider("rain", w.rain, 0, 1, &WeatherChange::rain);
            slider("fog density", w.fogDensity, 0, float(kMaxFogDensity), &WeatherChange::fogDensity,
                   ImGuiSliderFlags_Logarithmic, "%.5f /m");
            // The rendered density only while the rain raises it
            if (w.rain > 0)
            {
                const double factor = 1.0 + double(Vulkan::kRainFogFactor) * w.rain;
                DebugUiStyle::property("fog with rain");
                ImGui::Text("%.5f /m (x%.2f)", w.fogDensity * factor, factor);
            }
            slider("snow cover", w.snowCover, 0, 1, &WeatherChange::snowCover);
            DebugUiStyle::property("wind (x, z)");
            float wind[2] = {float(w.wind(0)), float(w.wind(1))};
            if (ImGui::DragFloat2("##wind", wind, 0.1f, -float(kMaxWind), float(kMaxWind), "%.1f m/s"))
            {
                WeatherChange c;
                c.wind = Vec2(wind[0], wind[1]);
                queueWeather(c);
            }
            // Round 2, finding 12: set_weather's other keys
            slider("wave amplitude", w.waveAmplitude, 0, float(kMaxWaveAmplitude), &WeatherChange::waveAmplitude,
                   ImGuiSliderFlags_Logarithmic, "%.2f m");
            slider("wave steepness", w.waveSteepness, 0, 1, &WeatherChange::waveSteepness);
            DebugUiStyle::property("cloud offset (x, z)");
            float offset[2] = {float(w.cloudOffset(0)), float(w.cloudOffset(1))};
            if (ImGui::DragFloat2("##cloud offset", offset, 10.0f, 0, 0, "%.0f m"))
            {
                WeatherChange c;
                c.cloudOffset = Vec2(offset[0], offset[1]);
                queueWeather(c);
            }
            DebugUiStyle::endProperties();
        }
        if (!writable) ImGui::EndDisabled();
    }
    if (!writable) DebugUiStyle::coloredText(DebugUiStyle::kRef, "a client: the host writes the time and weather");
}

void LightingTab::sunAndMoon()
{
    DebugUi& ui = app.debugUi();
    if (!ui.section("lighting/Sun and moon", "Sun and moon", false)) return;
    const EnvironmentSample e = app.sampleViewEnvironment();
    const CelestialState& c   = e.celestial;
    if (DebugUiStyle::beginProperties("##lighting bodies"))
    {
        DebugUiStyle::property("sun");
        ImGui::Text("elevation %.1f, azimuth %.1f deg", c.sunElevationDeg, c.sunAzimuthDeg);
        DebugUiStyle::property("sun illuminance");
        ImGui::Text("%.0f lux top, %.0f ground", double(e.state.sunIlluminance),
                    groundIlluminance(double(e.state.sunIlluminance), c.sunElevationDeg));
        DebugUiStyle::property("moon");
        ImGui::Text("elevation %.1f, azimuth %.1f deg", c.moonElevationDeg, c.moonAzimuthDeg);
        DebugUiStyle::property("moon phase");
        ImGui::Text("%.2f", c.moonPhase);
        DebugUiStyle::property("moon illuminance");
        ImGui::Text("%.3f lux top, %.3f ground", double(e.state.moonIlluminance),
                    groundIlluminance(double(e.state.moonIlluminance), c.moonElevationDeg));
        DebugUiStyle::property("shadow light");
        ImGui::TextUnformatted(shadowLightName(e.shadowLight));
        // The sun override (inspection; EnvironmentSettings::sunOverride)
        Vulkan::EnvironmentSettings s = app.environmentSettings();
        DebugUiStyle::property("override sun");
        bool overridden    = s.sunOverride.has_value();
        const bool toggled = ImGui::Checkbox("##override sun", &overridden);
        ui.markItem("lighting/override sun");
        if (toggled)
        {
            if (overridden)
                s.sunOverride = Vulkan::SunOverride{float(c.sunAzimuthDeg), float(c.sunElevationDeg)};
            else
                s.sunOverride.reset();
            applySettings(s);
        }
        if (s.sunOverride)
        {
            DebugUiStyle::property("azimuth");
            bool changed = ImGui::SliderFloat("##override azimuth", &s.sunOverride->azimuthDeg, 0, 360, "%.1f deg");
            DebugUiStyle::property("elevation");
            changed = ImGui::SliderFloat("##override elevation", &s.sunOverride->elevationDeg, -90, 90, "%.1f deg") ||
                      changed;
            if (changed) applySettings(s);
        }
        DebugUiStyle::endProperties();
    }
}

void LightingTab::exposure()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Exposure", "Exposure", false)) return;
    const EnvironmentSample e     = app.sampleViewEnvironment();
    Vulkan::EnvironmentSettings s = app.environmentSettings();
    if (DebugUiStyle::beginProperties("##lighting exposure"))
    {
        // The tone map (clay: the studio light; time of day has no effect); the same names as F3 Engine's
        DebugUiStyle::property("tone map");
        int tone            = scene->parameters().toneMap == Vulkan::ToneMap::Aces ? 1 : 0;
        const char* tones[] = {"clay (identity)", "ACES"};
        if (ImGui::Combo("##tone map", &tone, tones, 2))
            scene->setToneMap(tone == 1 ? Vulkan::ToneMap::Aces : Vulkan::ToneMap::Identity);
        DebugUiStyle::property("mode");
        int mode                = int(s.exposure.mode);
        const char* modeNames[] = {"time of day (curve)", "fixed"};
        if (ImGui::Combo("##exposure mode", &mode, modeNames, 2))
        {
            s.exposure.mode = Vulkan::ExposureMode(mode);
            applySettings(s);
        }
        // The EV100 of the fixed mode (set_lighting ev100's range); only in that mode (UI rating round 1, finding 11:
        // a disabled slider read like the current value)
        if (s.exposure.mode == Vulkan::ExposureMode::Fixed)
        {
            DebugUiStyle::property("fixed EV100");
            if (ImGui::SliderFloat("##exposure ev100", &s.exposure.ev100, -20, 30, "%.2f")) applySettings(s);
        }
        DebugUiStyle::property("EV100 now");
        if (scene->hasEnvironment())
            ImGui::Text("%.2f", double(scene->environment().ev100));
        else
            DebugUiStyle::coloredText(DebugUiStyle::kRef, "no environment (unit-less exposure)");
        DebugUiStyle::property("curve EV100");
        ImGui::Text("%.2f", double(e.curveEv100));
        DebugUiStyle::property("compensation");
        if (ImGui::SliderFloat("##exposure compensation", &s.exposure.compensation, -10, 10, "%.1f EV"))
            applySettings(s);
        // The frame's exposure: from the EV100 times F3's multiplier (UI rating round 1, finding 5: "exposure" on two
        // tabs with two meanings); the pre-exposure in the tooltip
        DebugUiStyle::property("effective exposure");
        ImGui::Text("%.3g (multiplier %.3g)", double(scene->frameExposure()), double(scene->exposure()));
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The frame's exposure: from the EV100, times the exposure multiplier of F3 Engine > "
                              "Rendering.\nPre-exposure (the scene pass's scale): %.3g",
                              double(scene->preExposure()));
        // The luminance of the frame: the Luminance preview and its reduction (the mapping of set_lighting_debug,
        // log2 by default; previewImage below shows it, the summary requests it)
        const Vulkan::PreviewMapping& mapping = app.lightingPreviewMapping();
        const Vulkan::LightingStatistics& ls  = scene->lightingStatistics();
        DebugUiStyle::property(mapping.log ? "luminance (log2)" : "luminance");
        if (ls.luminanceFrame > 0)
            ImGui::Text("mean %.2f  max %.2f  (%s)", double(ls.luminance.mean), double(ls.luminance.max),
                        scene->sceneWritesLinear() ? "cd/m^2" : "display");
        else
            ImGui::TextUnformatted("-");
        DebugUiStyle::endProperties();
    }
    // The plots with their labels above them (UI rating round 1, finding 12: the overlay text covered the data)
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    if (ls.luminanceFrame > 0)
    {
        ImGui::TextDisabled("luminance histogram (16 bins over the preview's range)");
        float bins[16];
        for (int i = 0; i < 16; ++i) bins[i] = float(ls.luminance.histogram[size_t(i)]);
        ImGui::PlotHistogram("##luminance histogram", bins, 16, 0, nullptr, 0.0f, FLT_MAX, ImVec2(-1, 48));
    }
    // The exposure curve with the current sun elevation (a marker on the curve)
    {
        const double elevation = e.celestial.sunElevationDeg;
        ImGui::TextDisabled("EV100 by sun elevation (-20 .. 90 deg), now %.0f deg", elevation);
        float curve[56];
        for (int i = 0; i < 56; ++i) curve[i] = s.exposure.curveEv100(-20.0f + 2.0f * float(i));
        const float lo = -4.0f, hi = 18.0f;
        ImGui::PlotLines("##exposure curve", curve, 56, 0, nullptr, lo, hi, ImVec2(-1, 48));
        // The plot's inner rectangle (the frame without its padding): x over -20 .. 90 deg, y over lo .. hi
        const ImVec2 pad = ImGui::GetStyle().FramePadding;
        const ImVec2 a(ImGui::GetItemRectMin().x + pad.x, ImGui::GetItemRectMin().y + pad.y);
        const ImVec2 b(ImGui::GetItemRectMax().x - pad.x, ImGui::GetItemRectMax().y - pad.y);
        const float el = float(std::clamp(elevation, -20.0, 90.0));
        const float x  = a.x + (b.x - a.x) * (el + 20.0f) / 110.0f;
        const float v  = std::clamp(s.exposure.curveEv100(el), lo, hi);
        const float y  = b.y - (b.y - a.y) * (v - lo) / (hi - lo);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddLine(ImVec2(x, a.y), ImVec2(x, b.y), DebugUiStyle::kRef);
        dl->AddCircleFilled(ImVec2(x, y), 3.5f, DebugUiStyle::kWarning);
    }
    // The luminance preview (the set_lighting_debug mapping as is)
    ImGui::TextDisabled("luminance");
    previewImage(Vulkan::LightingImage::Luminance, app.lightingPreviewMapping().log, 256.0f);
}

void LightingTab::ambient()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Ambient", "Ambient", false)) return;
    Vulkan::EnvironmentSettings s        = app.environmentSettings();
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    if (DebugUiStyle::beginProperties("##lighting ambient"))
    {
        DebugUiStyle::property("scale");
        bool changed = ImGui::SliderFloat("##ambient scale", &s.ambient.scale, 0, 4, "%.2f");
        DebugUiStyle::property("minimum RGB (lux)");
        changed = ImGui::DragFloat3("##min ambient", s.ambient.minAmbient.data(), 0.001f, 0, 100, "%.3f") || changed;
        if (changed) applySettings(s);
        DebugUiStyle::property("dominant (SH L0)");
        const vec3 d  = ls.sh[0];
        const float m = std::max({d(0), d(1), d(2), 1e-6f});
        ImGui::ColorButton("##dominant", ImVec4(d(0) / m, d(1) / m, d(2) / m, 1), ImGuiColorEditFlags_NoTooltip);
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::Text("%.3g  %.3g  %.3g lux", double(d(0)), double(d(1)), double(d(2)));
        DebugUiStyle::endProperties();
    }
    // The details in sub-sections (UI rating round 1, finding 6)
    if (ui.subSection("lighting/Ambient: SH coefficients", "SH coefficients (lux)", false) &&
        DebugUiStyle::beginProperties("##lighting sh"))
    {
        for (int i = 1; i < 9; ++i)
        {
            const std::string label = "SH " + std::to_string(i);
            DebugUiStyle::property(label.c_str());
            const vec3& c = ls.sh[size_t(i)];
            ImGui::Text("%.3g  %.3g  %.3g", double(c(0)), double(c(1)), double(c(2)));
        }
        DebugUiStyle::endProperties();
    }
    // The prototype albedo colours of the physical path (D21; not saved, S43 replaces them)
    if (ui.subSection("lighting/Ambient: Prototype albedo", "Prototype albedo (not saved)", false))
    {
        Vulkan::PrototypeSettings p = scene->prototypeSettings();
        Vulkan::PrototypeAlbedo& a  = p.colors;
        bool changed                = false;
        if (DebugUiStyle::beginProperties("##lighting albedo"))
        {
            auto color = [&](const char* label, vec3& c)
            {
                DebugUiStyle::property(label);
                changed = ImGui::ColorEdit3((std::string("##albedo ") + label).c_str(), c.data(),
                                            ImGuiColorEditFlags_Float) ||
                          changed;
            };
            color("ground", a.ground);
            color("rock", a.rock);
            color("shore", a.shore);
            color("water", a.water);
            color("grass", a.grass);
            color("leaves", a.leaves);
            color("bark", a.bark);
            DebugUiStyle::property("rock slope (n.y)");
            changed = ImGui::SliderFloat("##rock slope", &a.rockSlope, 0, 1, "%.2f") || changed;
            DebugUiStyle::endProperties();
        }
        if (changed) scene->setPrototypeSettings(p);
    }
    // The environment map (S36a1c): its capture progress and the front cube as a cross of a mip
    if (!ui.subSection("lighting/Ambient: Environment map", "Environment map", false)) return;
    const Vulkan::EnvironmentMapStatistics& em = ls.environmentMap;
    if (DebugUiStyle::beginProperties("##lighting environment"))
    {
        DebugUiStyle::property("state");
        if (em.valid)
            ImGui::Text("complete in frame %llu, %llu captures", (unsigned long long)em.lastCompleteFrame,
                        (unsigned long long)em.captures);
        else
            DebugUiStyle::coloredText(DebugUiStyle::kRef, "no capture yet (the physical path captures it)");
        DebugUiStyle::property("capture");
        if (em.progress > 0)
            ImGui::ProgressBar(em.steps > 0 ? float(em.progress) / float(em.steps) : 0.0f, ImVec2(-1, 0),
                               "capturing");
        else
            ImGui::TextUnformatted("idle");
        DebugUiStyle::property("preview mip");
        ImGui::SliderInt("##environment mip", &environmentMip, 0, Vulkan::EnvironmentMap::kMips - 1);
        // Round 2, finding 12: set_lighting's environment_resolution and environment_update_degrees
        Vulkan::LightingParameters lp = scene->lightingParameters();
        DebugUiStyle::property("resolution");
        int resolution = lp.environmentResolution;
        bool lpChanged = false;
        for (int r : {32, 64, 128, 256, 512})
        {
            if (r != 32) ImGui::SameLine();
            lpChanged = ImGui::RadioButton((std::to_string(r) + "##environment resolution").c_str(), &resolution, r) ||
                        lpChanged;
        }
        DebugUiStyle::property("update every");
        lpChanged = ImGui::SliderFloat("##environment update degrees", &lp.environmentUpdateDegrees, 0.01f, 90.0f,
                                       "%.2f deg of sun", ImGuiSliderFlags_Logarithmic) ||
                    lpChanged;
        lp.environmentResolution = resolution;
        if (lpChanged) applyParameters(lp);
        DebugUiStyle::endProperties();
    }
    ImGui::TextDisabled("the cube as a cross (log2), mip %d", environmentMip);
    previewImage(Vulkan::LightingImage::Environment, true, 384.0f, environmentMip);
}

void LightingTab::skyAndClouds()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Sky and clouds", "Sky and clouds", false)) return;
    Vulkan::EnvironmentSettings s        = app.environmentSettings();
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    bool changed                         = false;
    if (DebugUiStyle::beginProperties("##lighting sky"))
    {
        // Round 2, finding 12: set_lighting's clouds
        DebugUiStyle::property("cloud layer");
        Vulkan::LightingParameters lp = scene->lightingParameters();
        if (ImGui::Checkbox("##clouds", &lp.clouds)) applyParameters(lp);
        ImGui::SameLine();
        if (ls.clouds)
            ImGui::Text("mean coverage %.2f", double(ls.cloudMeanCoverage));
        else
            ImGui::TextDisabled("off this frame");
        if (!ls.clouds && ImGui::IsItemHovered())
            ImGui::SetTooltip("Clay, an enclosed space, clouds off or cloudiness 0");
        DebugUiStyle::property("cloud altitude");
        changed = ImGui::SliderFloat("##cloud altitude", &s.clouds.altitude, 200, 10000, "%.0f m") || changed;
        DebugUiStyle::property("cloud radius");
        changed = ImGui::SliderFloat("##cloud radius", &s.clouds.radius, 1000, 100000, "%.0f m") || changed;
        DebugUiStyle::property("cloud shadow fade");
        changed = ImGui::SliderFloat("##cloud shadow fade", &s.clouds.shadowFade, 100, 50000, "%.0f m") || changed;
        DebugUiStyle::property("cloud seed");
        int seed = int(std::min<uint32_t>(s.clouds.seed, uint32_t(Vulkan::kMaxCloudSeed)));
        if (ImGui::InputInt("##cloud seed", &seed))
        {
            s.clouds.seed = uint32_t(std::clamp(seed, 0, Vulkan::kMaxCloudSeed));
            changed       = true;
        }
        DebugUiStyle::property("fog falloff");
        changed = ImGui::SliderFloat("##fog falloff", &s.clouds.fogFalloff, 0.0001f, 0.01f, "%.5f /m",
                                     ImGuiSliderFlags_Logarithmic) ||
                  changed;
        DebugUiStyle::property("star intensity");
        changed = ImGui::SliderFloat("##star intensity", &s.starIntensity, 0, 10, "%.2f") || changed;
        DebugUiStyle::endProperties();
    }
    // The atmosphere and the previews in sub-sections (UI rating round 1, finding 6: five stacked previews)
    if (ui.subSection("lighting/Sky and clouds: Atmosphere", "Atmosphere", false) &&
        DebugUiStyle::beginProperties("##lighting atmosphere"))
    {
        DebugUiStyle::property("Rayleigh, Mie, ozone");
        float scales[3] = {s.atmosphere.rayleighScale, s.atmosphere.mieScale, s.atmosphere.ozoneScale};
        if (ImGui::DragFloat3("##atmosphere scales", scales, 0.01f, 0, 10, "x%.2f"))
        {
            s.atmosphere.rayleighScale = scales[0];
            s.atmosphere.mieScale      = scales[1];
            s.atmosphere.ozoneScale    = scales[2];
            changed                    = true;
        }
        DebugUiStyle::property("ground albedo");
        changed = ImGui::SliderFloat("##ground albedo", &s.atmosphere.groundAlbedo, 0, 1, "%.2f") || changed;
        DebugUiStyle::endProperties();
    }
    if (changed) applySettings(s);
    if (!ui.subSection("lighting/Sky and clouds: Previews", "Previews", false)) return;
    ImGui::TextDisabled("cloud coverage (the tile)");
    previewImage(Vulkan::LightingImage::CloudCoverage, false, 256.0f);
    ImGui::TextDisabled("cloud shadow (around the camera, +- the shadow fade)");
    previewImage(Vulkan::LightingImage::CloudShadow, false, 256.0f);
    ImGui::TextDisabled("transmittance LUT");
    previewImage(Vulkan::LightingImage::TransmittanceLut, false, 256.0f);
    ImGui::TextDisabled("sky-view LUT (log2 cd/m^2)");
    previewImage(Vulkan::LightingImage::SkyViewLut, true, 288.0f);
    ImGui::TextDisabled("multiple-scattering LUT");
    previewImage(Vulkan::LightingImage::MultiScatteringLut, false, 128.0f);
}

void LightingTab::cascadesSection()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Cascades", "Cascades", false)) return;
    Vulkan::LightingParameters p         = scene->lightingParameters();
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    bool changed                         = false;
    changed                              = ImGui::Checkbox("sun shadows", &p.sunShadows) || changed;
    ImGui::SameLine();
    changed = ImGui::Checkbox("terrain casts", &p.terrainCastsInCascades) || changed;
    ImGui::SameLine();
    // S36a2: freeze the cascades (with the cascade bounds and frustum splits overlays: what each cascade covers); the
    // summary's tag shows a frozen state
    bool frozen = scene->cascadesFrozen();
    if (ImGui::Checkbox("freeze", &frozen)) scene->setFreezeCascades(frozen);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("The cascades keep the current camera (the overlays show them)");
    // The state of the frame (the light's direction in the tooltip; the summary has its elevation)
    const char* light = ls.shadowLight == 0   ? "none"
                        : !ls.physical        ? "studio sun"
                        : ls.shadowLight == 1 ? "sun"
                                              : "moon";
    if (ls.shadows)
    {
        ImGui::PushTextWrapPos(0);
        ImGui::Text("%d cascades from the %s, %s filter, %.1f MiB", ls.cascadeCount, light,
                    ls.linearShadowFilter ? "linear" : "manual bilinear", double(ls.shadowMemoryBytes) / (1 << 20));
        ImGui::PopTextWrapPos();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("light direction (%.2f, %.2f, %.2f)", double(ls.shadowLightDir(0)),
                              double(ls.shadowLightDir(1)), double(ls.shadowLightDir(2)));
    }
    else
        DebugUiStyle::coloredText(
            DebugUiStyle::kRef,
            p.sunShadows ? "no cascades this frame (no shadow light or an enclosed space)" : "sun shadows off");
    // The GPU time against the 2 ms of the measurement rule (the shadow budget is in GPU and memory)
    {
        Profiler& prof  = app.gpuProfile();
        const double ms = prof.scopeStats(prof.scope("gpu/shadow.cascades"), LightingGpuTimes::kMeanFrames).avg;
        char text[96];
        std::snprintf(text, sizeof(text), "shadow.cascades %.3f ms%s (mean of %d frames)", ms,
                      ms > 2.0 ? ", above 2 ms" : "", LightingGpuTimes::kMeanFrames);
        DebugUiStyle::coloredText(ms > 2.0 ? DebugUiStyle::kWarn : DebugUiStyle::kGpu, text);
    }
    if (ls.shadows)
    {
        // A row per cascade (the casters of its last render); a horizontal scroll with the index column frozen (UI
        // rating round 1, finding 9: the terrain column was cut), numbers right-aligned
        const ImGuiStyle& style = ImGui::GetStyle();
        const float height      = float(ls.cascadeCount + 1) * (ImGui::GetTextLineHeight() + 2 * style.CellPadding.y) +
                             style.ScrollbarSize + 4;
        if (ImGui::BeginTable("##cascade rows", 8,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX,
                              ImVec2(0, height)))
        {
            ImGui::TableSetupScrollFreeze(1, 1);
            for (const char* h : {"#", "split m", "texel m", "every", "updated", "instances", "triangles", "terrain"})
                ImGui::TableSetupColumn(h);
            ImGui::TableHeadersRow();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("terrain: CDLOD nodes (deduplicated with the previous cascade), triangles");
            for (int c = 0; c < ls.cascadeCount; ++c)
            {
                const Vulkan::CascadeStatistics& k = ls.cascades[size_t(c)];
                const vec3 col                     = Vulkan::cascadeColor(c);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(ImVec4(col(0), col(1), col(2), 1), "%d", c);
                cellRightf("%.1f - %.0f", double(k.splitNear), double(k.splitFar));
                cellRightf("%.3f", double(k.texelSize));
                cellRightf("%d", k.interval);
                cellRightf("%llu", (unsigned long long)k.lastUpdate);
                cellRightf("%u", k.instances);
                cellRightf("%.1fk", double(k.triangles) / 1000);
                cellRightf("%u (%u) %.0fk", k.terrainNodes, k.terrainDeduplicated, double(k.terrainTriangles) / 1000);
            }
            ImGui::EndTable();
        }
        // The cascades' depth side by side; a click enlarges one (again: back to the strip)
        if (enlargedCascade < 0 || enlargedCascade >= ls.cascadeCount)
        {
            enlargedCascade = -1;
            ImGui::TextDisabled("depth (light space, 0 near .. 1 far); click a cascade to enlarge it");
            ImVec2 at;
            if (previewImageClicked(Vulkan::LightingImage::CascadeStrip, false, 512.0f, at))
                enlargedCascade = std::clamp(int(at.x * float(ls.cascadeCount)), 0, ls.cascadeCount - 1);
        }
        else
        {
            ImGui::TextDisabled("cascade %d (texel %.3f m, half extent %.0f m); click to go back", enlargedCascade,
                                double(ls.cascades[size_t(enlargedCascade)].texelSize),
                                double(ls.cascades[size_t(enlargedCascade)].halfExtent));
            ImVec2 at;
            if (previewImageClicked(Vulkan::LightingImage::Cascade, false, 512.0f, at, 0, enlargedCascade))
                enlargedCascade = -1;
        }
    }
    // The parameters in sub-sections below what is looked at (UI rating round 1, finding 6: 14 sliders came first)
    if (ui.subSection("lighting/Cascades: Split and resolution", "Split and resolution", false) &&
        DebugUiStyle::beginProperties("##lighting cascade split"))
    {
        DebugUiStyle::property("count");
        changed = ImGui::SliderInt("##cascade count", &p.cascadeCount, 1, 4) || changed;
        DebugUiStyle::property("resolution");
        {
            const int sizes[]   = {512, 1024, 2048, 4096};
            const char* names[] = {"512", "1024", "2048", "4096"};
            int current         = 2;
            for (int i = 0; i < 4; ++i)
                if (sizes[i] == p.cascadeResolution) current = i;
            if (ImGui::Combo("##cascade resolution", &current, names, 4))
            {
                p.cascadeResolution = sizes[current];
                changed             = true;
            }
        }
        DebugUiStyle::property("distance");
        changed = ImGui::SliderFloat("##shadow distance", &p.shadowDistance, 50, 10000, "%.0f m",
                                     ImGuiSliderFlags_Logarithmic) ||
                  changed;
        DebugUiStyle::property("split lambda");
        changed = ImGui::SliderFloat("##split lambda", &p.splitLambda, 0, 1, "%.2f") || changed;
        DebugUiStyle::property("blend band");
        changed = ImGui::SliderFloat("##cascade blend", &p.cascadeBlend, 0, 0.5f, "%.2f") || changed;
        DebugUiStyle::property("update every 1-3");
        if (ImGui::InputInt3("##cascade intervals", &p.cascadeInterval[1]))
        {
            for (int i = 1; i < 4; ++i) p.cascadeInterval[size_t(i)] = std::clamp(p.cascadeInterval[size_t(i)], 1, 8);
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cascades 1-3 render every n-th frame (cascade 0 every frame)");
        DebugUiStyle::endProperties();
    }
    if (ui.subSection("lighting/Cascades: Bias", "Bias", false) &&
        DebugUiStyle::beginProperties("##lighting cascade bias"))
    {
        DebugUiStyle::property("depth (texels)");
        changed = ImGui::SliderFloat("##depth bias", &p.depthBiasTexels, 0, 8, "%.2f") || changed;
        DebugUiStyle::property("slope");
        changed = ImGui::SliderFloat("##slope bias", &p.slopeBias, 0, 8, "%.2f") || changed;
        DebugUiStyle::property("normal offset (texels)");
        changed = ImGui::SliderFloat("##normal offset", &p.normalOffsetTexels, 0, 8, "%.2f") || changed;
        DebugUiStyle::endProperties();
    }
    if (ui.subSection("lighting/Cascades: Casters", "Casters", false) &&
        DebugUiStyle::beginProperties("##lighting cascade casters"))
    {
        DebugUiStyle::property("LOD bias (cascades 2-3)");
        changed = ImGui::SliderInt("##shadow lod bias", &p.shadowLodBias, 0, 3) || changed;
        DebugUiStyle::property("smallest (texels)");
        changed = ImGui::SliderFloat("##small casters", &p.smallCasterTexels, 0, 10, "%.1f") || changed;
        DebugUiStyle::property("max height");
        changed = ImGui::SliderFloat("##max caster height", &p.maxCasterHeight, 0, 1000, "%.0f m") || changed;
        DebugUiStyle::property("min light elevation");
        changed =
            ImGui::SliderFloat("##min shadow elevation", &p.minShadowElevationDeg, 0.5f, 45, "%.1f deg") || changed;
        DebugUiStyle::property("terrain distance");
        changed =
            ImGui::SliderFloat("##terrain caster distance", &p.cascadeCasterDistance, 0, 5000, "%.0f m") || changed;
        DebugUiStyle::endProperties();
    }
    if (changed) applyParameters(p);
}

void LightingTab::horizonSection()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Far terrain shadows", "Far terrain shadows", false)) return;
    Vulkan::LightingParameters p       = scene->lightingParameters();
    const Vulkan::HorizonStatistics& h = scene->lightingStatistics().horizon;
    bool changed                       = ImGui::Checkbox("horizon shadows", &p.horizonShadows);
    ImGui::SameLine();
    if (ImGui::SmallButton("recompute now")) scene->requestHorizonRecompute(false);
    if (DebugUiStyle::beginProperties("##lighting horizon"))
    {
        DebugUiStyle::property("resolution");
        {
            const int sizes[]   = {256, 512, 1024, 2048};
            const char* names[] = {"256", "512", "1024", "2048"};
            int current         = 2;
            for (int i = 0; i < 4; ++i)
                if (sizes[i] == p.horizonResolution) current = i;
            if (ImGui::Combo("##horizon resolution", &current, names, 4))
            {
                p.horizonResolution = sizes[current];
                changed             = true;
            }
        }
        DebugUiStyle::property("extent");
        changed = ImGui::SliderFloat("##horizon extent", &p.horizonExtent, 4096, 131072, "%.0f m",
                                     ImGuiSliderFlags_Logarithmic) ||
                  changed;
        DebugUiStyle::property("rows per frame");
        changed =
            ImGui::SliderInt("##horizon rows", &p.horizonRowsPerFrame, 1, 1024, "%d", ImGuiSliderFlags_Logarithmic) ||
            changed;
        DebugUiStyle::property("map");
        if (h.valid)
            ImGui::Text("%d^2, %.1f m texels, %.0f m", h.resolution, h.texel, h.extent);
        else
            ImGui::TextUnformatted(p.horizonShadows ? "none yet (no terrain or shadow light)" : "off");
        DebugUiStyle::property("anchor");
        ImGui::Text("(%d, %d) m, reference %.0f m", h.corner(0), h.corner(1), h.reference);
        DebugUiStyle::property("light direction");
        ImGui::Text("(%.3f, %.3f, %.3f), el %.2f deg", double(h.light(0)), double(h.light(1)), double(h.light(2)),
                    std::asin(std::clamp(double(h.light(1)), -1.0, 1.0)) * kDegrees);
        // A bar only while a map is computed (UI rating round 1, finding 8: idle showed a full bar with 0 rows)
        DebugUiStyle::property("progress");
        if (h.computing)
        {
            char text[64];
            std::snprintf(text, sizeof(text), "%d / %d rows", h.rowsDone, h.resolution);
            ImGui::ProgressBar(h.resolution > 0 ? float(h.rowsDone) / float(h.resolution) : 0.0f, ImVec2(-1, 0), text);
        }
        else if (h.completed > 0)
            ImGui::Text("idle (last map complete in frame %llu)", (unsigned long long)h.lastComplete);
        else
            ImGui::TextUnformatted("idle");
        DebugUiStyle::property("blend to the new map");
        ImGui::Text("%.0f %%", double(h.blend) * 100);
        DebugUiStyle::property("rows this frame");
        ImGui::Text("%d", h.rowsThisFrame);
        DebugUiStyle::property("recomputes");
        ImGui::Text("%llu started, %llu completed", (unsigned long long)h.recomputes, (unsigned long long)h.completed);
        DebugUiStyle::property("march level at 10 km");
        ImGui::Text("%d (%.0f m cells)", h.levelAt10km, 32.0 * double(1 << h.levelAt10km));
        DebugUiStyle::endProperties();
    }
    if (changed) applyParameters(p);
    if (!h.valid) return;
    ImGui::RadioButton("R (from 64 m)", &horizonChannel, 0);
    ImGui::SameLine();
    ImGui::RadioButton("G (beyond the cascades' terrain)", &horizonChannel, 1);
    ImGui::TextDisabled("shadow height (m above the reference; light: high)");
    previewImage(Vulkan::LightingImage::Horizon, false, 384.0f, 0, horizonChannel);
}

void LightingTab::localLightsSection()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Local lights", "Local lights", false)) return;
    Vulkan::LightingParameters p          = scene->lightingParameters();
    const Vulkan::LightingStatistics& ls  = scene->lightingStatistics();
    const Vulkan::LocalLightStatistics& c = ls.lights;
    bool changed                          = ImGui::Checkbox("local lights", &p.localLights);
    ImGui::SameLine();
    changed = ImGui::Checkbox("clustered", &p.clusteredLights) || changed;
    ImGui::SameLine();
    overlayCheckbox(*scene, Vulkan::OverlayLightVolumes);
    if (DebugUiStyle::beginProperties("##lighting lights"))
    {
        DebugUiStyle::property("max lights");
        changed =
            ImGui::SliderInt("##max lights", &p.maxLights, 1, 4096, "%d", ImGuiSliderFlags_Logarithmic) || changed;
        ui.markItem("lighting/max lights");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("at least 1: the \"local lights\" checkbox switches them off");
        // (UI rating round 1, finding 10: one line of four counts was cut)
        DebugUiStyle::property("visible");
        ImGui::Text("%u of %u (%u candidates)", c.visible, c.submitted, c.candidates);
        DebugUiStyle::property("over budget");
        char text[32];
        std::snprintf(text, sizeof(text), "%u", c.overBudget);
        DebugUiStyle::coloredText(c.overBudget > 0 ? DebugUiStyle::kWarn : DebugUiStyle::kWhite, text);
        DebugUiStyle::property("shadowed");
        ImGui::Text("%u, %u without atlas tiles", c.shadowed, c.withoutTiles);
        DebugUiStyle::property("light buffers");
        ImGui::Text("%.1f KiB", double(c.memoryBytes) / 1024);
        DebugUiStyle::endProperties();
    }
    if (changed) applyParameters(p);
    if (!p.localLights)
        DebugUiStyle::coloredText(DebugUiStyle::kRef, "local lights are off (LightingParameters::localLights)");
    else if (scene->parameters().toneMap == Vulkan::ToneMap::Identity)
        DebugUiStyle::coloredText(DebugUiStyle::kRef, "clay: the local lights are not shaded (studio light)");
    // The light table and the selected light's editor (the clusters and the shadow atlas are sections of their own: UI
    // rating round 1, finding 4)
    lightTable();
}

void LightingTab::clustersSection()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Light clusters", "Light clusters", false)) return;
    Vulkan::LightingParameters p          = scene->lightingParameters();
    const Vulkan::LocalLightStatistics& c = scene->lightingStatistics().lights;
    Vulkan::LightingDebugState d          = scene->lightingDebug();
    if (!p.localLights) DebugUiStyle::coloredText(DebugUiStyle::kRef, "local lights are off (Local lights)");
    bool changed = false;
    // The clusters (B3): the grid, the occupancy, the heat map
    if (DebugUiStyle::beginProperties("##lighting clusters"))
    {
        DebugUiStyle::property("grid (x, y, z)");
        int grid[3] = {p.clusterX, p.clusterY, p.clusterZ};
        if (ImGui::InputInt3("##cluster grid", grid))
        {
            p.clusterX = std::clamp(grid[0], 1, 64);
            p.clusterY = std::clamp(grid[1], 1, 64);
            p.clusterZ = std::clamp(grid[2], 2, 128);
            changed    = true;
        }
        DebugUiStyle::property("near / far");
        float depth[2] = {p.clusterNear, p.clusterFar};
        if (ImGui::InputFloat2("##cluster depth", depth, "%.1f m"))
        {
            p.clusterFar  = std::max(depth[1], 1.0f);
            p.clusterNear = std::clamp(depth[0], 0.01f, p.clusterFar * 0.99f);
            changed       = true;
        }
        DebugUiStyle::property("slots per cluster");
        changed =
            ImGui::SliderInt("##cluster slots", &p.maxLightsPerCluster, 1, 1024, "%d", ImGuiSliderFlags_Logarithmic) ||
            changed;
        DebugUiStyle::property("occupancy");
        ImGui::Text("max %u, mean %.2f", c.clusterMax, c.clusterMean);
        DebugUiStyle::property("overflowed");
        char text[64];
        std::snprintf(text, sizeof(text), "%u clusters, %llu lights dropped", c.overflowedClusters,
                      (unsigned long long)c.droppedLights);
        DebugUiStyle::coloredText(c.overflowedClusters > 0 ? DebugUiStyle::kWarn : DebugUiStyle::kWhite, text);
        if (c.overflowedClusters > 0 && ImGui::IsItemHovered())
            ImGui::SetTooltip("An overflowed cluster drops its least important lights");
        DebugUiStyle::property("heat scale");
        float scale = scene->clusterHeatScaleValue();
        if (ImGui::SliderFloat("##heat scale", &scale, 1, 256, "%.0f lights", ImGuiSliderFlags_Logarithmic))
            scene->setClusterHeatScale(scale);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The lights per cluster at the top of the heat map's colors");
        DebugUiStyle::endProperties();
    }
    if (changed) applyParameters(p);
    // The cluster heat map: the view (the same label as in the view combo) and the preview (a slice or the maximum
    // over the slices)
    bool heat = d.view == Vulkan::LightingDebugView::ClusterLights;
    if (ImGui::Checkbox(kViewLabels[size_t(Vulkan::LightingDebugView::ClusterLights)], &heat))
    {
        d.view = heat ? Vulkan::LightingDebugView::ClusterLights : Vulkan::LightingDebugView::None;
        scene->setLightingDebug(d);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s\n(view cluster_lights)", kViewTooltips[size_t(Vulkan::LightingDebugView::ClusterLights)]);
    ImGui::SameLine();
    overlayCheckbox(*scene, Vulkan::OverlayClusterGrid);
    if (c.visible == 0) return;
    const float scale = scene->clusterHeatScaleValue();
    const int slices  = scene->frameClusterGrid().z;
    ImGui::SetNextItemWidth(10 * ImGui::GetFontSize());
    ImGui::SliderInt("preview slice", &clusterSlice, -1, slices - 1, clusterSlice < 0 ? "max of all" : "%d");
    clusterSlice                       = std::clamp(clusterSlice, -1, slices - 1);
    Vulkan::PreviewMapping m           = app.lightingPreviewMapping();
    m.autoRange                        = false;
    m.min                              = 0;
    m.max                              = scale;
    m.colorMap                         = Vulkan::ColorMap::Turbo;
    const Vulkan::PreviewMapping saved = app.lightingPreviewMapping();
    app.lightingPreviewMapping()       = m;
    ImGui::TextDisabled("lights per cluster (turbo, 0 .. %.0f)", double(scale));
    Selection sel;
    sel.slice = clusterSlice;
    previewImage(Vulkan::LightingImage::Clusters, false, 384.0f, 0, 0, sel);
    app.lightingPreviewMapping() = saved;
}

void LightingTab::atlasSection()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Shadow atlas", "Shadow atlas", false)) return;
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    Vulkan::LightingDebugState d         = scene->lightingDebug();
    // The shadow atlas (B4): its parameters, the cache, the preview of a layer with the tiles' owners (a click on a
    // tile selects its light)
    const Vulkan::AtlasStatistics& as = ls.atlas;
    Vulkan::LightingParameters ap     = scene->lightingParameters();
    if (!ap.localLights) DebugUiStyle::coloredText(DebugUiStyle::kRef, "local lights are off (Local lights)");
    bool atlasChanged = false;
    if (DebugUiStyle::beginProperties("##lighting atlas"))
    {
        DebugUiStyle::property("atlas / tile");
        {
            // The combos show the stored values (CodeRabbit #86: an admitted value outside the lists, e.g. 512 or 64,
            // showed another entry); every admitted power of two is listed
            auto combo = [](const char* id, int& value, const std::vector<int>& choices)
            {
                const std::string preview = std::to_string(value);
                bool changed              = false;
                ImGui::SetNextItemWidth(80);
                if (ImGui::BeginCombo(id, preview.c_str()))
                {
                    for (int c : choices)
                        if (ImGui::Selectable(std::to_string(c).c_str(), c == value))
                        {
                            changed = c != value;
                            value   = c;
                        }
                    ImGui::EndCombo();
                }
                return changed;
            };
            if (combo("##atlas resolution", ap.atlasResolution, {512, 1024, 2048, 4096, 8192}))
            {
                ap.atlasTile = std::min(ap.atlasTile, ap.atlasResolution);
                atlasChanged = true;
            }
            ImGui::SameLine();
            std::vector<int> tileChoices;
            for (int t = 64; t <= std::min(ap.atlasResolution, 1024); t *= 2) tileChoices.push_back(t);
            atlasChanged = combo("##atlas tile", ap.atlasTile, tileChoices) || atlasChanged;
        }
        DebugUiStyle::property("faces per frame");
        atlasChanged = ImGui::SliderInt("##faces per frame", &ap.facesPerFrame, 0, 11) || atlasChanged;
        DebugUiStyle::property("depth bias (16-bit steps)");
        atlasChanged = ImGui::SliderFloat("##atlas bias", &ap.atlasDepthBias, 0, 64, "%.1f") || atlasChanged;
        DebugUiStyle::property("normal offset (texels)");
        atlasChanged = ImGui::SliderFloat("##atlas normal", &ap.atlasNormalOffset, 0, 8, "%.2f") || atlasChanged;
        DebugUiStyle::property("tiles");
        ImGui::Text("%u / %u used (%d texels)", as.tilesUsed, as.tilesTotal, as.tile);
        DebugUiStyle::property("cache this frame");
        ImGui::Text("%u hits, %u static + %u dynamic renders, %u clears", as.hits, as.staticRenders,
                    as.dynamicRenders, as.dynamicClears);
        DebugUiStyle::property("queue");
        ImGui::Text("%u static faces waiting, %u dynamic faces skipped", as.queue, as.dynamicSkipped);
        DebugUiStyle::property("terrain nodes");
        ImGui::Text("%u (%u cut by the 16384 limit)", as.terrainNodes, as.terrainNodesTruncated);
        DebugUiStyle::property("shadowed lights");
        ImGui::Text("%u with tiles, %u without (the tile budget)", as.lightsWithTiles, as.lightsWithoutTiles);
        DebugUiStyle::property("atlas memory");
        ImGui::Text("%.1f MiB", double(as.memoryBytes) / (1 << 20));
        DebugUiStyle::endProperties();
    }
    if (atlasChanged) applyParameters(ap);
    int longRange = 0;
    for (const Vulkan::LocalLights::Light& l : scene->localLights().lights())
        longRange += l.data.castShadows && l.data.range > 100 ? 1 : 0;
    if (longRange > 0)
    {
        char text[96];
        std::snprintf(text, sizeof(text), "%d shadowed lights with a range above 100 m (coarse 16-bit depth)",
                      longRange);
        DebugUiStyle::coloredText(DebugUiStyle::kWarn, text);
    }
    if (!as.on) return;
    int& layer = atlasLayer;
    ImGui::RadioButton("static layer", &layer, 0);
    ImGui::SameLine();
    ImGui::RadioButton("dynamic layer", &layer, 1);
    ImGui::SameLine();
    overlayCheckbox(*scene, Vulkan::OverlayAtlasOwners);
    d = scene->lightingDebug();
    ImGui::PushTextWrapPos(0);
    ImGui::TextDisabled("linear depth (distance / range; white: nothing); click a tile to select its light");
    ImGui::PopTextWrapPos();
    ImVec2 at;
    const bool clicked = previewImageClicked(Vulkan::LightingImage::Atlas, false, 384.0f, at, 0, layer);
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    const Vulkan::ShadowAtlas& atlas = scene->shadowAtlas();
    const int perSide                = std::max(atlas.tilesPerSide(), 1);
    if (clicked)
    {
        const int tx   = std::clamp(int(at.x * float(perSide)), 0, perSide - 1);
        const int ty   = std::clamp(int(at.y * float(perSide)), 0, perSide - 1);
        const size_t t = size_t(ty * perSide + tx);
        if (t < atlas.tiles().size() && atlas.tiles()[t].owner)
        {
            d.highlightLight = atlas.tiles()[t].owner;
            scene->setLightingDebug(d);
        }
    }
    if (scene->lightingDebugImages().preview(Vulkan::LightingImage::Atlas) && (d.overlays & Vulkan::OverlayAtlasOwners))
    {
        // The owners: a frame per used tile (the selected light's yellow), its light and face
        ImDrawList* dl   = ImGui::GetWindowDrawList();
        const float cell = (hi.x - lo.x) / float(perSide);
        for (int t = 0; t < int(atlas.tiles().size()); ++t)
        {
            const Vulkan::AtlasTileInfo& ti = atlas.tiles()[size_t(t)];
            if (!ti.owner) continue;
            const ImVec2 a(lo.x + float(t % perSide) * cell, lo.y + float(t / perSide) * cell);
            const ImVec2 b(a.x + cell, a.y + cell);
            const bool sel  = ti.owner == d.highlightLight;
            const ImU32 col = sel              ? kSelectedLight
                              : ti.staticValid ? IM_COL32(80, 220, 120, 200)
                                               : IM_COL32(230, 90, 70, 220);
            dl->AddRect(ImVec2(a.x + 1, a.y + 1), ImVec2(b.x - 1, b.y - 1), col, 0, 0, sel ? 2.0f : 1.0f);
            char label[24];
            std::snprintf(label, sizeof(label), "%u:%d%s", ti.owner, ti.face, ti.dynamicContent ? "d" : "");
            if (cell >= 28) dl->AddText(ImVec2(a.x + 3, a.y + 2), col, label);
        }
    }
    // The selected light's tile at full resolution
    const std::vector<int> selTiles = d.highlightLight ? atlas.tilesOf(d.highlightLight) : std::vector<int>();
    if (!selTiles.empty())
    {
        atlasTileFace = std::clamp(atlasTileFace, 0, int(selTiles.size()) - 1);
        if (selTiles.size() > 1)
        {
            ImGui::SetNextItemWidth(140);
            ImGui::SliderInt("face (+x -x +y -y +z -z)", &atlasTileFace, 0, int(selTiles.size()) - 1);
        }
        ImGui::TextDisabled("light %u, tile %d", d.highlightLight, selTiles[size_t(atlasTileFace)]);
        Selection sel;
        sel.light = d.highlightLight;
        sel.face  = atlasTileFace;
        previewImage(Vulkan::LightingImage::AtlasTile, false, 256.0f, 0, layer, sel);
    }
}

void LightingTab::lightTable()
{
    Vulkan::RenderScene* scene   = app.scene();
    Vulkan::LightingDebugState d = scene->lightingDebug();
    // The light table: every light of the scene (the frame's ones with their buffer index) and the entity lights the
    // scene does not have (off, invalid, no pose: greyed rows with their state; round 1, finding 8: a light switched
    // off disappeared and could not be switched on again); a row selects the light (an entity light by its entity)
    const Vulkan::LocalLights& table                                = scene->localLights();
    const std::vector<ExtractSystem::LightEntityState> entityLights = app.world().extract().lightStates();
    std::unordered_map<uint32_t, Entity> entities;
    std::unordered_map<uint32_t, const ExtractSystem::LightEntityState*> byEntity;
    for (const ExtractSystem::LightEntityState& st : entityLights)
    {
        if (st.handle) entities[st.handle.id] = st.entity;
        byEntity[st.entity.index] = &st;
    }
    // An explicit clear (MCP highlight_light: null) drops the selection, also of an entity light that is off (its
    // highlight was 0 already)
    if (app.lightHighlightClears() != seenHighlightClears)
    {
        seenHighlightClears = app.lightHighlightClears();
        if (d.highlightLight == 0) selectedEntity = Entity();
    }
    // Another change of the highlight (an atlas tile click, MCP) selects that light's entity
    if (d.highlightLight != tabHighlight)
    {
        auto e         = entities.find(d.highlightLight);
        selectedEntity = e != entities.end() ? e->second : Entity();
        tabHighlight   = d.highlightLight;
    }
    const ExtractSystem::LightEntityState* selState = nullptr;
    if (selectedEntity.valid())
    {
        auto it = byEntity.find(selectedEntity.index);
        if (it != byEntity.end() && it->second->entity == selectedEntity)
            selState = it->second;
        else
            selectedEntity = Entity();
    }
    // The selected entity light's current sink light is the highlight (0 while it is not rendered)
    if (selState && selState->handle.id != d.highlightLight)
    {
        d.highlightLight = selState->handle.id;
        tabHighlight     = d.highlightLight;
        scene->setLightingDebug(d);
    }
    std::unordered_map<uint32_t, const Vulkan::FrameLightInfo*> frame;
    for (const Vulkan::FrameLightInfo& f : scene->frameLights()) frame[f.handle] = &f;
    struct Row
    {
        const Vulkan::LocalLights::Light* light;       // null: an entity light that is not rendered
        const Vulkan::FrameLightInfo* info;            // null: not in this frame
        const ExtractSystem::LightEntityState* state;  // null: a scene light
        double distance;
        uint64_t key;
    };
    auto worldOf = [](const ExtractSystem::LightEntityState& st)
    { return Vec3(st.pose.position + st.pose.rotation * (st.source.offset * st.pose.scale)); };
    std::vector<Row> rows;
    const Vec3 camera = app.viewCamera().position;
    size_t unrendered = 0;
    for (const Vulkan::LocalLights::Light& l : table.lights())
    {
        auto f  = frame.find(l.handle.id);
        auto e  = entities.find(l.handle.id);
        auto st = e != entities.end() ? byEntity.find(e->second.index) : byEntity.end();
        rows.push_back({&l, f == frame.end() ? nullptr : f->second, st != byEntity.end() ? st->second : nullptr,
                        (l.worldPosition() - camera).norm(), uint64_t(l.handle.id)});
    }
    for (const ExtractSystem::LightEntityState& st : entityLights)
        if (!st.handle)
        {
            unrendered++;
            rows.push_back({nullptr, nullptr, &st,
                            st.hasPose ? (worldOf(st) - camera).norm() : std::numeric_limits<double>::infinity(),
                            (uint64_t(1) << 32) | st.entity.index});
        }
    // (UI rating round 1, finding 11: "(0 in lists)" and "0 entity lights not rendered" were always shown)
    {
        std::string caption = std::to_string(table.size()) + " lights, " + std::to_string(frame.size()) + " visible";
        if (table.listCount() > 0) caption += ", " + std::to_string(table.listCount()) + " from lists";
        if (unrendered > 0) caption += ", " + std::to_string(unrendered) + " entity lights not rendered";
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(caption.c_str());
        ImGui::PopTextWrapPos();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("The visible lights first, then the other scene lights, then the entity lights that are "
                              "not rendered (greyed); a row selects its light");
    }
    // Columns clipped in a narrow tab (round 1, finding 17): a horizontal scroll with the handle column frozen, the
    // units in the headers
    const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                  ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit;
    const float height = ImGui::GetTextLineHeightWithSpacing() * float(std::min<size_t>(rows.size(), 12) + 3);
    // (round 2, finding 12: A8's buffer index, the light's index in the frame's light buffer, as the last column)
    if (!rows.empty() && ImGui::BeginTable("##light table", 11, flags, ImVec2(0, height)))
    {
        ImGui::TableSetupScrollFreeze(1, 1);
        const char* headers[] = {"handle",  "entity",   "state", "kind",         "distance m", "lumen",
                                 "range m", "clusters", "tiles", "shadow frame", "buffer"};
        for (int i = 0; i < 11; ++i)
            ImGui::TableSetupColumn(headers[i], i == 4 ? ImGuiTableColumnFlags_DefaultSort : 0);
        ImGui::TableHeadersRow();
        // The selected row in the selected light's yellow (UI rating round 1, finding 11: hardly visible)
        ImGui::PushStyleColor(ImGuiCol_Header, (kSelectedLight & 0x00FFFFFFu) | (90u << IM_COL32_A_SHIFT));
        if (ImGuiTableSortSpecs* sort = ImGui::TableGetSortSpecs())
            if (sort->SpecsCount > 0)
            {
                lightSortColumn    = sort->Specs[0].ColumnIndex;
                lightSortAscending = sort->Specs[0].SortDirection != ImGuiSortDirection_Descending;
            }
        auto key = [&](const Row& r) -> double
        {
            const bool spot    = r.light ? r.light->data.kind == Vulkan::LightKind::Spot
                                         : r.state->source.kind == Vulkan::LightKind::Spot;
            const double lumen = r.light ? double(r.light->data.lumen) : r.state->source.lumen;
            const double range = r.light ? double(r.light->data.range) : r.state->source.range;
            switch (lightSortColumn)
            {
                case 0:
                    return r.light ? double(r.light->handle.id) : 4294967296.0;
                case 1:
                    return r.state ? double(r.state->entity.index) : -1.0;
                case 2:
                    return r.light ? (r.info ? 0.0 : 1.0) : 2.0;
                case 3:
                    return spot ? 1.0 : 0.0;
                case 5:
                    return lumen;
                case 6:
                    return range;
                case 7:
                    return r.info ? double(r.info->clusters) : -1.0;
                case 8:
                    return r.info ? double(r.info->tiles) : -1.0;
                case 9:
                    return r.info ? double(r.info->lastRender) : -1.0;
                case 10:
                    return r.info ? double(r.info->index) : -1.0;
                default:
                    return r.distance;
            }
        };
        // The frame's lights, the other scene lights, the unrendered ones
        auto rowGroup = [](const Row& r) { return r.info ? 0 : (r.light ? 1 : 2); };
        // Sorted when the sort spec or the light set changes, else every 30 frames (the distances change slowly; S30's
        // thousands of piece lights)
        const uint64_t now = scene->frames();
        bool resort        = orderColumn != lightSortColumn || orderAscending != lightSortAscending ||
                      lightOrder.size() != rows.size() || now >= orderFrame + 30 || now < orderFrame;
        if (!resort)
        {
            std::unordered_map<uint64_t, size_t> at;
            for (size_t k = 0; k < rows.size(); ++k) at[rows[k].key] = k;
            std::vector<Row> ordered;
            ordered.reserve(rows.size());
            for (uint64_t k : lightOrder)
            {
                auto it = at.find(k);
                if (it == at.end()) break;
                ordered.push_back(rows[it->second]);
            }
            if (ordered.size() == rows.size())
                rows.swap(ordered);
            else
                resort = true;
            // A light that changed its group (visible, rendered, not rendered) moves at once (CodeRabbit #86)
            for (size_t k = 1; k < rows.size() && !resort; ++k) resort = rowGroup(rows[k]) < rowGroup(rows[k - 1]);
        }
        if (resort)
        {
            std::stable_sort(rows.begin(), rows.end(),
                             [&](const Row& a, const Row& b)
                             {
                                 // The frame's lights first, then the other scene lights, then the unrendered ones
                                 const int ga = rowGroup(a), gb = rowGroup(b);
                                 if (ga != gb) return ga < gb;
                                 // A NaN key (an invalid light's lumen) after the numbers in both directions (a
                                 // strict weak order; CodeRabbit #86)
                                 const double ka = key(a), kb = key(b);
                                 const bool na = std::isnan(ka), nb = std::isnan(kb);
                                 if (na != nb) return nb;
                                 if (!na && ka != kb) return lightSortAscending ? ka < kb : ka > kb;
                                 return a.key < b.key;
                             });
            lightOrder.clear();
            for (const Row& r : rows) lightOrder.push_back(r.key);
            orderColumn    = lightSortColumn;
            orderAscending = lightSortAscending;
            orderFrame     = now;
        }
        ImGuiListClipper clipper;
        clipper.Begin(int(rows.size()));
        while (clipper.Step())
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
            {
                const Row& r = rows[size_t(i)];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                char label[48];
                if (r.light)
                    std::snprintf(label, sizeof(label), "%u", r.light->handle.id);
                else
                    std::snprintf(label, sizeof(label), "-##e%u", r.state->entity.index);
                const bool selected = r.state ? r.state->entity == selectedEntity
                                              : !selectedEntity.valid() && d.highlightLight == r.light->handle.id;
                if (!r.info) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (ImGui::Selectable(label, selected, ImGuiSelectableFlags_SpanAllColumns))
                {
                    selectedEntity   = selected || !r.state ? Entity() : r.state->entity;
                    d.highlightLight = selected ? 0u : (r.light ? r.light->handle.id : 0u);
                    tabHighlight     = d.highlightLight;
                    scene->setLightingDebug(d);
                }
                ImGui::TableNextColumn();
                // (the generation only when it is not 0)
                if (r.state && r.state->entity.generation != 0)
                    ImGui::Text("%u:%u", r.state->entity.index, r.state->entity.generation);
                else if (r.state)
                    ImGui::Text("%u", r.state->entity.index);
                else
                    ImGui::TextUnformatted(r.light->list ? "list" : "-");
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.light ? (r.info ? "visible" : "rendered") : r.state->state.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(Vulkan::lightKindName(r.light ? r.light->data.kind : r.state->source.kind));
                // The numbers right-aligned
                if (std::isfinite(r.distance))
                    cellRightf("%.1f", r.distance);
                else
                    DebugUiStyle::cellRight("-");
                cellRightf("%.0f", r.light ? double(r.light->data.lumen) : r.state->source.lumen);
                cellRightf("%.1f", r.light ? double(r.light->data.range) : r.state->source.range);
                if (r.info)
                    cellRightf("%u", r.info->clusters);
                else
                    DebugUiStyle::cellRight("-");
                const bool shadows = r.light ? r.light->data.castShadows : r.state->source.castShadows;
                if (r.info && r.info->tiles > 0)
                    cellRightf("%d%s", r.info->tiles, r.info->shadowReady ? "" : " (pending)");
                else
                    DebugUiStyle::cellRight(shadows ? (r.info ? "none" : "-") : "-");
                if (r.info && r.info->lastRender)
                    cellRightf("%llu", (unsigned long long)r.info->lastRender);
                else
                    DebugUiStyle::cellRight("-");
                if (r.info)
                    cellRightf("%u", r.info->index);
                else
                    DebugUiStyle::cellRight("-");
                if (!r.info) ImGui::PopStyleColor();
            }
        ImGui::PopStyleColor();
        ImGui::EndTable();
    }
    // The selected light: its numbers, the camera, an entity light's fields (written through set_component; also
    // while it is off or invalid)
    const Vulkan::LocalLights::Light* sel =
        d.highlightLight ? table.find(Vulkan::LightHandle{d.highlightLight}) : nullptr;
    if (!sel && !selState) return;
    Vec3 pos      = Vec3::Zero();
    Vec3 dir      = Vec3(0, -1, 0);
    double range  = 10;
    bool hasPos   = true;
    SpaceId space = 0;
    bool spot     = false;
    if (sel)
    {
        const Vulkan::LightData& ld = sel->data;
        pos                         = sel->worldPosition();
        dir                         = ld.direction.cast<double>();
        range                       = double(ld.range);
        space                       = sel->chunk.space;
        spot                        = ld.kind == Vulkan::LightKind::Spot;
        ImGui::PushTextWrapPos(0);
        ImGui::Text("light %u: %s at (%.1f, %.1f, %.1f), %.0f lm, range %.1f m%s", sel->handle.id,
                    Vulkan::lightKindName(ld.kind), pos(0), pos(1), pos(2), double(ld.lumen), double(ld.range),
                    ld.castShadows ? ", shadowed" : "");
        ImGui::PopTextWrapPos();
        if (ld.castShadows && ld.range > 100)
            DebugUiStyle::coloredText(DebugUiStyle::kWarn, "shadowed with a range above 100 m: coarse 16-bit depth");
    }
    else
    {
        hasPos = selState->hasPose;
        pos    = worldOf(*selState);
        // An invalid light (e.g. a NaN offset) is moved to at its entity's origin; without a finite one the camera is
        // not moved (CodeRabbit #86 round 2: the tool held the camera at a NaN eye)
        if (!pos.allFinite()) pos = selState->pose.position;
        hasPos = hasPos && pos.allFinite();
        dir    = selState->pose.rotation * selState->source.unitRotation() * Vec3(0, 0, -1);
        range  = selState->source.range;
        space  = selState->pose.space;
        spot   = selState->source.kind == Vulkan::LightKind::Spot;
        ImGui::Text("entity %u:%u: %s, not rendered (%s)", selectedEntity.index, selectedEntity.generation,
                    Vulkan::lightKindName(selState->source.kind), selState->state.c_str());
        if (selState->state.rfind("invalid", 0) == 0)
            DebugUiStyle::coloredText(DebugUiStyle::kWarn, "an invalid LightSource is not rendered");
    }
    const bool moveCamera = hasPos && ImGui::SmallButton("move camera to light");
    if (hasPos) app.debugUi().markItem("lighting/move camera to light");
    if (moveCamera)
    {
        const auto [eye, rotation] = lightViewPose(pos, spot ? dir : Vec3::Zero(), range, app.viewCamera().rotation);
        // Regression (S36 demo): the camera was held as by an MCP tool, so the free camera no longer moved; the button
        // moves it once and leaves it free (also from a tool's hold: the user takes the camera)
        app.releaseCamera();
        app.setViewCameraByTool(space, eye, rotation, app.viewCamera().fovy, false);
    }
    if (hasPos) ImGui::SameLine();
    if (ImGui::SmallButton("deselect"))
    {
        selectedEntity   = Entity();
        d.highlightLight = 0;
        tabHighlight     = 0;
        scene->setLightingDebug(d);
    }
    if (!selState)
    {
        ImGui::TextUnformatted(sel && sel->list ? "a scene light of a list (read-only)" : "a scene light (read-only)");
        return;
    }
    const Entity e = selectedEntity;
    World& w       = app.world();
    if (!w.alive(e) || !w.has<LightSource>(e)) return;
    // --coop (round 2, finding 10): the client's LightSource is replicated (HostReliable), its writes are refused; the
    // editor writes the host's entity of the same NetId (as the environment edits write the host's world)
    Entity target = e;
    if (app.coop())
    {
        const NetId id = w.replication().netIdOf(e);
        target         = id ? app.hostWorld()->replication().entityOfNetId(id) : Entity();
        if (!app.hostWorld()->alive(target) || !app.hostWorld()->has<LightSource>(target))
        {
            ImGui::TextUnformatted("read-only in --coop (no host entity of this light)");
            return;
        }
    }
    // The values of the entity that is written (round 4, minor 2: in --coop the client's replica lags the host by the
    // replication delay; a drag restarted from it each frame and an edit resent other fields' stale values)
    const World& owner       = app.coop() ? *app.hostWorld() : w;
    const LightSource before = owner.read<LightSource>(target);
    LightSource s            = before;
    bool edit                = false;
    // What is changed most often always, the rest in a sub-section (UI rating round 1, finding 6)
    if (DebugUiStyle::beginProperties("##lighting light source"))
    {
        DebugUiStyle::property("on");
        edit = ImGui::Checkbox("##light on", &s.on) || edit;
        app.debugUi().markItem("lighting/light on");
        // (UI rating round 1, finding 10: the hint was a cut line above the editor)
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("A game's systems may switch the light again (e.g. game_world's lamps follow the time "
                              "of day)");
        DebugUiStyle::property("cast shadows");
        edit = ImGui::Checkbox("##light shadows", &s.castShadows) || edit;
        app.debugUi().markItem("lighting/light shadows");
        DebugUiStyle::property("kind");
        int kind = s.kind == Vulkan::LightKind::Spot ? 1 : 0;
        if (ImGui::Combo("##light kind", &kind, "point\0spot\0"))
        {
            s.kind = kind == 1 ? Vulkan::LightKind::Spot : Vulkan::LightKind::Point;
            edit   = true;
        }
        DebugUiStyle::property("lumen");
        float lumen = float(s.lumen);
        if (ImGui::SliderFloat("##light lumen", &lumen, 0, 100000, "%.0f", ImGuiSliderFlags_Logarithmic))
        {
            s.lumen = lumen;
            edit    = true;
        }
        DebugUiStyle::property("range");
        float rangeF = float(s.range);
        if (ImGui::SliderFloat("##light range", &rangeF, 0.1f, 1000, "%.1f m", ImGuiSliderFlags_Logarithmic))
        {
            s.range = rangeF;
            edit    = true;
        }
        DebugUiStyle::property("color");
        float col[3] = {float(s.color(0)), float(s.color(1)), float(s.color(2))};
        if (ImGui::ColorEdit3("##light colour", col, ImGuiColorEditFlags_Float))
        {
            s.color = Vec3(col[0], col[1], col[2]);
            edit    = true;
        }
        DebugUiStyle::endProperties();
    }
    if (app.debugUi().subSection("lighting/Local lights: Light details", "Offset, spot, priority, flicker", false) &&
        DebugUiStyle::beginProperties("##lighting light details"))
    {
        DebugUiStyle::property("offset (entity, m)");
        float off[3] = {float(s.offset(0)), float(s.offset(1)), float(s.offset(2))};
        if (ImGui::DragFloat3("##light offset", off, 0.05f, -100, 100, "%.2f"))
        {
            s.offset = Vec3(off[0], off[1], off[2]);
            edit     = true;
        }
        if (s.kind == Vulkan::LightKind::Spot)
        {
            DebugUiStyle::property("cone (inner, outer)");
            float cone[2] = {float(s.innerAngleDeg), float(s.outerAngleDeg)};
            if (ImGui::SliderFloat2("##light cone", cone, 1, 89, "%.0f deg"))
            {
                s.innerAngleDeg = std::min(cone[0], cone[1]);
                s.outerAngleDeg = std::max(cone[0], cone[1]);
                edit            = true;
            }
            // The spot's direction relative to the entity (its -z after rotation) as yaw (about +y, 0: -z) and pitch
            DebugUiStyle::property("direction (yaw, pitch)");
            const Vec3 local = s.rotation * Vec3(0, 0, -1);
            float yp[2]      = {float(std::atan2(-local(0), -local(2)) * 57.29577951308232),
                                float(std::asin(std::clamp(local(1), -1.0, 1.0)) * 57.29577951308232)};
            if (ImGui::SliderFloat2("##light direction", yp, -180, 180, "%.0f deg"))
            {
                const double yaw   = double(yp[0]) / 57.29577951308232;
                const double pitch = std::clamp(double(yp[1]), -89.9, 89.9) / 57.29577951308232;
                const Vec3 f(-std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch));
                s.rotation = cameraLookRotation(f);
                edit       = true;
            }
        }
        DebugUiStyle::property("priority");
        float prio = float(s.priority);
        if (ImGui::SliderFloat("##light priority", &prio, 0, 10, "%.2f"))
        {
            s.priority = prio;
            edit       = true;
        }
        DebugUiStyle::property("flicker");
        float flicker[2] = {float(s.flickerAmplitude), float(s.flickerFrequency)};
        if (ImGui::SliderFloat2("##light flicker", flicker, 0, 10, "%.2f"))
        {
            s.flickerAmplitude = std::clamp(double(flicker[0]), 0.0, 1.0);
            s.flickerFrequency = flicker[1];
            edit               = true;
        }
        DebugUiStyle::endProperties();
    }
    if (edit)
    {
        // A write through the world API (set_component, logged; refused on a pure client), between the steps; only the
        // fields this frame changed (a field written meanwhile by the host or a system keeps its value)
        auto vec3 = [](const Vec3& v) { return json11::Json::array{v(0), v(1), v(2)}; };
        json11::Json::object values;
        if (s.on != before.on) values["on"] = s.on;
        if (s.castShadows != before.castShadows) values["castShadows"] = s.castShadows;
        if (s.kind != before.kind) values["kind"] = s.kind == Vulkan::LightKind::Spot ? "spot" : "point";
        if (s.lumen != before.lumen) values["lumen"] = s.lumen;
        if (s.range != before.range) values["range"] = s.range;
        if (s.color != before.color) values["color"] = vec3(s.color);
        if (s.offset != before.offset) values["offset"] = vec3(s.offset);
        if (s.rotation.coeffs() != before.rotation.coeffs())
            values["rotation"] = json11::Json::array{s.rotation.w(), s.rotation.x(), s.rotation.y(), s.rotation.z()};
        if (s.innerAngleDeg != before.innerAngleDeg) values["innerAngleDeg"] = s.innerAngleDeg;
        if (s.outerAngleDeg != before.outerAngleDeg) values["outerAngleDeg"] = s.outerAngleDeg;
        if (s.priority != before.priority) values["priority"] = s.priority;
        if (s.flickerAmplitude != before.flickerAmplitude) values["flickerAmplitude"] = s.flickerAmplitude;
        if (s.flickerFrequency != before.flickerFrequency) values["flickerFrequency"] = s.flickerFrequency;
        if (values.empty()) return;
        EngineApp* a = &app;
        app.queueBetweenSteps(
            [a, target, values]()
            {
                const json11::Json args = json11::Json::object{
                    {"entity", WorldApi::entityJson(target)}, {"component", "LightSource"}, {"values", values}};
                ApiResult r;
                if (a->coop())
                {
                    // The host's world (its API: the same validation and log line)
                    WorldApi host(*a->hostWorld());
                    r = host.call("set_component", args);
                }
                else
                    r = a->api().call("set_component", args);
                if (!r.ok) SAIGA_LOG("lighting", "set_component LightSource: %s", r.error.c_str());
            });
    }
}
void LightingTab::gpuAndMemory()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/GPU and memory", "GPU and memory", false)) return;
    // The lighting sections against the shadow budget, the debug sections' cost separately (get_lighting's list and
    // sums: lightingGpuTimes) in one table like F2's GPU sections: a sub-section indented under its parent, the ms
    // right-aligned (UI rating round 1, finding 11: "(120 frame mean)" on every row)
    const LightingGpuTimes t = lightingGpuTimes(app);
    ImGui::TextDisabled("GPU time, mean of %d frames", LightingGpuTimes::kMeanFrames);
    if (ImGui::BeginTable("##lighting gpu", 2, ImGuiTableFlags_RowBg))
    {
        ImGui::TableSetupColumn("section", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthFixed, 5 * ImGui::GetFontSize());
        DebugUiStyle::headerRow({"section", "ms"});
        auto row = [](const std::string& name, bool indent, double ms, ImU32 color)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (indent) ImGui::Indent();
            ImGui::TextUnformatted(name.c_str());
            if (indent) ImGui::Unindent();
            char text[32];
            std::snprintf(text, sizeof(text), "%.3f", ms);
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            DebugUiStyle::cellRight(text);
            ImGui::PopStyleColor();
        };
        for (bool debug : {false, true})
        {
            std::string parent;
            for (const LightingGpuTimes::Section& s : t.sections)
            {
                if (s.debug != debug) continue;
                // A sub-section ("shadow.cascades" under "shadow") by its name after the parent's
                const bool sub = !parent.empty() && s.name.rfind(parent + ".", 0) == 0;
                row(sub ? s.name.substr(parent.size() + 1) : s.name, sub, s.meanMs, DebugUiStyle::kGpu);
                if (!s.summed) parent = s.name;
            }
            if (!debug)
            {
                char name[64];
                std::snprintf(name, sizeof(name), "lighting (budget %.1f ms)", LightingGpuTimes::kBudgetMs);
                row(name, false, t.lightingMs,
                    t.lightingMs > LightingGpuTimes::kBudgetMs ? DebugUiStyle::kWarn : DebugUiStyle::kWhite);
            }
            else
                row("debug (not in the budget)", false, t.debugMs, DebugUiStyle::kWhite);
        }
        ImGui::EndTable();
    }
    if (DebugUiStyle::beginProperties("##lighting memory"))
    {
        const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
        DebugUiStyle::property("memory");
        ImGui::Text("%.2f MiB", double(ls.memoryBytes) / (1 << 20));
        DebugUiStyle::property("LUT recomputations");
        ImGui::Text("%llu", (unsigned long long)ls.lutComputations);
        DebugUiStyle::property("preview dispatches");
        ImGui::Text("%u this frame", ls.previewDispatches);
        // A performance switch (UI rating round 1, finding 3: it was in the toolbar)
        DebugUiStyle::property("force full updates");
        Vulkan::LightingParameters p = scene->lightingParameters();
        if (ImGui::Checkbox("##force full updates", &p.forceFullUpdates)) applyParameters(p);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Every incremental effect updates fully every frame (the goldens' mode)");
        DebugUiStyle::endProperties();
    }
}

}  // namespace Engine
}  // namespace Saiga
