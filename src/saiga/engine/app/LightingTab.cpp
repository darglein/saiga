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
#include "saiga/engine/core/Log.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/scene/EnvironmentMap.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
namespace
{
// A rough display colour of a body's light from its elevation (warmer near the horizon; the swatches only)
ImVec4 bodyColor(double elevationDeg, bool moon)
{
    if (elevationDeg <= 0) return ImVec4(0.1f, 0.1f, 0.15f, 1);
    if (moon) return ImVec4(0.7f, 0.75f, 0.9f, 1);
    const float t = float(std::min(elevationDeg / 30.0, 1.0));
    return ImVec4(1.0f, 0.55f + 0.4f * t, 0.3f + 0.6f * t, 1);
}

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
}  // namespace

LightingTab::~LightingTab()
{
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || scene->imguiRenderer() != previewImgui) return;
    for (ImTextureID t : previewTextures)
        if (t) scene->imguiRenderer()->removeTexture(t);
}

void LightingTab::draw()
{
    toolbar();
    timeAndWeather();
    sunAndMoon();
    exposure();
    ambient();
    skyAndClouds();
    gpuAndMemory();
}

void LightingTab::previewImage(Vulkan::LightingImage image, bool log, float maxWidth)
{
    Vulkan::RenderScene* scene     = app.scene();
    Vulkan::PreviewMapping mapping = app.lightingPreviewMapping();
    mapping.log                    = log;  // the luminance passes the set_lighting_debug mapping's log
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
            return;
        }
    }
    scene->lightingDebugImages().show(image, mapping);
    Vulkan::ImGuiVulkanRenderer* imgui = scene->imguiRenderer();
    if (!imgui || !preview) return;
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
    ImGui::Image(previewTextures[k], ImVec2(w, w * float(preview->height) / float(std::max(preview->width, 1u))));
}

void LightingTab::toolbar()
{
    Vulkan::RenderScene* scene = app.scene();
    if (!scene) return;
    // The tone map (clay: the studio light; time of day has no effect)
    int tone            = scene->parameters().toneMap == Vulkan::ToneMap::Aces ? 1 : 0;
    const char* tones[] = {"clay", "ACES"};
    ImGui::SetNextItemWidth(70);
    if (ImGui::Combo("##tone map", &tone, tones, 2))
        scene->setToneMap(tone == 1 ? Vulkan::ToneMap::Aces : Vulkan::ToneMap::Identity);
    ImGui::SameLine();
    Vulkan::LightingDebugState d = scene->lightingDebug();
    int view                     = int(d.view);
    std::vector<const char*> views;
    for (int i = 0; i < int(Vulkan::LightingDebugView::Count); ++i)
        views.push_back(Vulkan::lightingDebugViewName(Vulkan::LightingDebugView(i)));
    ImGui::SetNextItemWidth(130);
    // The views are drawn by the LIGHTING_PBR variant: without it a view would show clay (a1b-a1c)
    const bool variant = scene->lightingParameters().pbr;
    if (!variant) ImGui::BeginDisabled();
    if (ImGui::Combo("##debug view", &view, views.data(), int(views.size())))
    {
        d.view = Vulkan::LightingDebugView(view);
        scene->setLightingDebug(d);
    }
    if (!variant) ImGui::EndDisabled();
    for (const auto& o : Vulkan::lightingOverlayNames())
    {
        ImGui::SameLine();
        bool on = (d.overlays & o.second) != 0;
        if (ImGui::Checkbox(o.first, &on))
        {
            d.overlays = on ? (d.overlays | o.second) : (d.overlays & ~o.second);
            scene->setLightingDebug(d);
        }
    }
    Vulkan::LightingParameters p = scene->lightingParameters();
    bool changed                 = ImGui::Checkbox("force full updates", &p.forceFullUpdates);
    ImGui::SameLine();
    // a1b-a1c: the LIGHTING_PBR pipeline variant (the debug views need it)
    changed = ImGui::Checkbox("PBR variant", &p.pbr) || changed;
    if (changed) scene->setLightingParameters(p);
    ImGui::SameLine();
    if (ImGui::SmallButton("copy JSON"))
    {
        json11::Json out;
        std::string error;
        if (lightingJson(app, {}, out, error)) ImGui::SetClipboardText(out.dump().c_str());
    }
    if (scene->parameters().toneMap == Vulkan::ToneMap::Identity)
        DebugUiStyle::coloredText(DebugUiStyle::kRef, "clay: studio light, time of day has no effect");
    if (!p.pbr) DebugUiStyle::coloredText(DebugUiStyle::kRef, "the debug views need the PBR variant (until S36a1c)");
    if (d.view == Vulkan::LightingDebugView::Exposure) exposureLegend();
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
        DebugUiStyle::property("rate");
        ImGui::Text("x%.0f", e.time.rate);
        for (double r : {0.0, 1.0, 60.0, 600.0})
        {
            ImGui::SameLine();
            const std::string label = "x" + std::to_string(int(r));
            if (ImGui::SmallButton(label.c_str())) queueTime(WorldTimeChange{std::nullopt, std::nullopt, r});
            ui.markItem("lighting/rate " + label);
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
    if (hasWeather(target) && DebugUiStyle::beginProperties("##lighting weather"))
    {
        const Weather& w = e.weather;
        auto slider = [&](const char* label, double value, float lo, float hi, std::optional<double> WeatherChange::* f,
                          ImGuiSliderFlags flags = 0, const char* format = "%.3f")
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
               ImGuiSliderFlags_Logarithmic, "%.5f");
        slider("snow cover", w.snowCover, 0, 1, &WeatherChange::snowCover);
        DebugUiStyle::property("wind (x, z)");
        float wind[2] = {float(w.wind(0)), float(w.wind(1))};
        if (ImGui::DragFloat2("##wind", wind, 0.1f, -float(kMaxWind), float(kMaxWind), "%.1f m/s"))
        {
            WeatherChange c;
            c.wind = Vec2(wind[0], wind[1]);
            queueWeather(c);
        }
        DebugUiStyle::property("space");
        ImGui::Text("%d%s", int(space), e.enclosed ? " (enclosed)" : "");
        DebugUiStyle::endProperties();
    }
    if (!writable) ImGui::EndDisabled();
    if (!writable) DebugUiStyle::coloredText(DebugUiStyle::kRef, "a client: the host writes the time and weather");
}

void LightingTab::sunAndMoon()
{
    DebugUi& ui = app.debugUi();
    if (!ui.section("lighting/Sun and moon", "Sun and moon", true)) return;
    const EnvironmentSample e = app.sampleViewEnvironment();
    const CelestialState& c   = e.celestial;
    if (DebugUiStyle::beginProperties("##lighting bodies"))
    {
        DebugUiStyle::property("sun");
        ImGui::ColorButton("##sun color", bodyColor(c.sunElevationDeg, false), ImGuiColorEditFlags_NoTooltip,
                           ImVec2(12, 12));
        ImGui::SameLine();
        ImGui::Text("el %.1f  az %.1f deg", c.sunElevationDeg, c.sunAzimuthDeg);
        DebugUiStyle::property("sun illuminance");
        ImGui::Text("%.0f lux top, %.0f ground", double(e.state.sunIlluminance),
                    groundIlluminance(double(e.state.sunIlluminance), c.sunElevationDeg));
        DebugUiStyle::property("moon");
        ImGui::ColorButton("##moon color", bodyColor(c.moonElevationDeg, true), ImGuiColorEditFlags_NoTooltip,
                           ImVec2(12, 12));
        ImGui::SameLine();
        ImGui::Text("el %.1f  az %.1f deg  phase %.2f", c.moonElevationDeg, c.moonAzimuthDeg, c.moonPhase);
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
            app.setEnvironmentSettings(s);
        }
        if (s.sunOverride)
        {
            DebugUiStyle::property("azimuth");
            bool changed = ImGui::SliderFloat("##override azimuth", &s.sunOverride->azimuthDeg, 0, 360, "%.1f deg");
            DebugUiStyle::property("elevation");
            changed = ImGui::SliderFloat("##override elevation", &s.sunOverride->elevationDeg, -90, 90, "%.1f deg") ||
                      changed;
            if (changed) app.setEnvironmentSettings(s);
        }
        DebugUiStyle::endProperties();
    }
}

void LightingTab::exposure()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Exposure", "Exposure", true)) return;
    const EnvironmentSample e     = app.sampleViewEnvironment();
    Vulkan::EnvironmentSettings s = app.environmentSettings();
    if (DebugUiStyle::beginProperties("##lighting exposure"))
    {
        DebugUiStyle::property("mode");
        int mode                = int(s.exposure.mode);
        const char* modeNames[] = {"time of day (curve)", "fixed"};
        if (ImGui::Combo("##exposure mode", &mode, modeNames, 2))
        {
            s.exposure.mode = Vulkan::ExposureMode(mode);
            app.setEnvironmentSettings(s);
        }
        // The EV100 of the fixed mode (set_lighting ev100's range)
        DebugUiStyle::property("fixed EV100");
        const bool fixed = s.exposure.mode == Vulkan::ExposureMode::Fixed;
        if (!fixed) ImGui::BeginDisabled();
        if (ImGui::SliderFloat("##exposure ev100", &s.exposure.ev100, -20, 30, "%.2f")) app.setEnvironmentSettings(s);
        if (!fixed) ImGui::EndDisabled();
        DebugUiStyle::property("EV100 now");
        if (scene->hasEnvironment())
            ImGui::Text("%.2f", double(scene->environment().ev100));
        else
            DebugUiStyle::coloredText(DebugUiStyle::kRef, "no environment (unit-less exposure)");
        DebugUiStyle::property("EV100 of the curve");
        ImGui::Text("%.2f", double(e.curveEv100));
        DebugUiStyle::property("compensation");
        if (ImGui::SliderFloat("##exposure compensation", &s.exposure.compensation, -10, 10, "%.1f EV"))
            app.setEnvironmentSettings(s);
        DebugUiStyle::property("exposure");
        ImGui::Text("%.3g (multiplier %.3g)", double(scene->frameExposure()), double(scene->exposure()));
        DebugUiStyle::property("pre-exposure");
        ImGui::Text("%.3g", double(scene->preExposure()));
        // The luminance of the frame: the Luminance preview and its reduction while the section is open (the mapping
        // of set_lighting_debug, log2 by default; previewImage below shows it)
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
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    if (ls.luminanceFrame > 0)
    {
        float bins[16];
        for (int i = 0; i < 16; ++i) bins[i] = float(ls.luminance.histogram[size_t(i)]);
        ImGui::PlotHistogram("##luminance histogram", bins, 16, 0, "luminance histogram", 0.0f, FLT_MAX,
                             ImVec2(-1, 48));
    }
    // The exposure curve with the current sun elevation
    {
        float curve[56];
        for (int i = 0; i < 56; ++i) curve[i] = s.exposure.curveEv100(-20.0f + 2.0f * float(i));
        const std::string label = "EV100 by elevation (-20..90), now " +
                                  std::to_string(int(std::lround(e.celestial.sunElevationDeg))) + " deg";
        ImGui::PlotLines("##exposure curve", curve, 56, 0, label.c_str(), -4.0f, 18.0f, ImVec2(-1, 48));
    }
    // The luminance preview (the set_lighting_debug mapping as is)
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
        DebugUiStyle::property("minimum (lux)");
        changed = ImGui::DragFloat3("##min ambient", s.ambient.minAmbient.data(), 0.001f, 0, 100, "%.3f") || changed;
        if (changed) app.setEnvironmentSettings(s);
        DebugUiStyle::property("dominant (SH L0)");
        const vec3 d  = ls.sh[0];
        const float m = std::max({d(0), d(1), d(2), 1e-6f});
        ImGui::ColorButton("##dominant", ImVec4(d(0) / m, d(1) / m, d(2) / m, 1), ImGuiColorEditFlags_NoTooltip,
                           ImVec2(12, 12));
        ImGui::SameLine();
        ImGui::Text("%.4g %.4g %.4g lux", double(d(0)), double(d(1)), double(d(2)));
        for (int i = 1; i < 9; ++i)
        {
            const std::string label = "SH " + std::to_string(i);
            DebugUiStyle::property(label.c_str());
            const vec3& c = ls.sh[size_t(i)];
            ImGui::Text("%.4g %.4g %.4g", double(c(0)), double(c(1)), double(c(2)));
        }
        DebugUiStyle::endProperties();
    }
    // The prototype albedo colours of the physical path (D21; not saved, S43 replaces them)
    Vulkan::PrototypeSettings p = scene->prototypeSettings();
    Vulkan::PrototypeAlbedo& a  = p.colors;
    bool changed                = false;
    if (DebugUiStyle::beginProperties("##lighting albedo"))
    {
        auto color = [&](const char* label, vec3& c)
        {
            DebugUiStyle::property(label);
            changed =
                ImGui::ColorEdit3((std::string("##albedo ") + label).c_str(), c.data(), ImGuiColorEditFlags_Float) ||
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
    // The environment map (S36a1c): its capture progress and the front cube as a cross of a mip
    const Vulkan::EnvironmentMapStatistics& em = ls.environmentMap;
    if (DebugUiStyle::beginProperties("##lighting environment"))
    {
        DebugUiStyle::property("environment map");
        if (em.valid)
            ImGui::Text("complete in frame %llu, %llu captures", (unsigned long long)em.lastCompleteFrame,
                        (unsigned long long)em.captures);
        else
            DebugUiStyle::coloredText(DebugUiStyle::kRef, "no capture yet (the physical path captures it)");
        DebugUiStyle::property("capture progress");
        ImGui::ProgressBar(em.steps > 0 ? float(em.progress) / float(em.steps) : 0.0f, ImVec2(-1, 0),
                           em.progress > 0 ? "capturing" : "idle");
        DebugUiStyle::property("mip");
        ImGui::SliderInt("##environment mip", &environmentMip, 0, Vulkan::EnvironmentMap::kMips - 1);
        DebugUiStyle::endProperties();
    }
    scene->setEnvironmentPreviewMip(environmentMip);
    previewImage(Vulkan::LightingImage::Environment, true, 384.0f);
}

void LightingTab::skyAndClouds()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/Sky and clouds", "Sky and clouds", false)) return;
    Vulkan::EnvironmentSettings s = app.environmentSettings();
    bool changed                  = false;
    if (DebugUiStyle::beginProperties("##lighting sky"))
    {
        DebugUiStyle::property("cloud altitude");
        changed = ImGui::SliderFloat("##cloud altitude", &s.clouds.altitude, 200, 10000, "%.0f m") || changed;
        DebugUiStyle::property("cloud radius");
        changed = ImGui::SliderFloat("##cloud radius", &s.clouds.radius, 1000, 100000, "%.0f m") || changed;
        DebugUiStyle::property("cloud shadow fade");
        changed = ImGui::SliderFloat("##cloud shadow fade", &s.clouds.shadowFade, 100, 50000, "%.0f m") || changed;
        DebugUiStyle::property("fog falloff");
        changed = ImGui::SliderFloat("##fog falloff", &s.clouds.fogFalloff, 0.0001f, 0.01f, "%.5f /m",
                                     ImGuiSliderFlags_Logarithmic) ||
                  changed;
        DebugUiStyle::property("star intensity");
        changed = ImGui::SliderFloat("##star intensity", &s.starIntensity, 0, 10, "%.2f") || changed;
        DebugUiStyle::property("Rayleigh / Mie / ozone");
        float scales[3] = {s.atmosphere.rayleighScale, s.atmosphere.mieScale, s.atmosphere.ozoneScale};
        if (ImGui::DragFloat3("##atmosphere scales", scales, 0.01f, 0, 10, "%.2f"))
        {
            s.atmosphere.rayleighScale = scales[0];
            s.atmosphere.mieScale      = scales[1];
            s.atmosphere.ozoneScale    = scales[2];
            changed                    = true;
        }
        DebugUiStyle::property("ground albedo");
        changed = ImGui::SliderFloat("##ground albedo", &s.atmosphere.groundAlbedo, 0, 1, "%.2f") || changed;
        DebugUiStyle::property("LUT recomputations");
        ImGui::Text("%llu", (unsigned long long)scene->lightingStatistics().lutComputations);
        DebugUiStyle::endProperties();
    }
    if (changed) app.setEnvironmentSettings(s);
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    if (ls.clouds)
        ImGui::Text("cloud layer on, mean coverage %.2f", double(ls.cloudMeanCoverage));
    else
        DebugUiStyle::coloredText(DebugUiStyle::kRef, "cloud layer off (clay, enclosed, clouds false or cloudiness 0)");
    ImGui::TextUnformatted("cloud coverage (the tile)");
    previewImage(Vulkan::LightingImage::CloudCoverage, false, 256.0f);
    ImGui::TextUnformatted("cloud shadow (around the camera, +- the shadow fade)");
    previewImage(Vulkan::LightingImage::CloudShadow, false, 256.0f);
    ImGui::TextUnformatted("transmittance LUT");
    previewImage(Vulkan::LightingImage::TransmittanceLut, false, 256.0f);
    ImGui::TextUnformatted("sky-view LUT (log2 cd/m^2)");
    previewImage(Vulkan::LightingImage::SkyViewLut, true, 288.0f);
}

void LightingTab::gpuAndMemory()
{
    DebugUi& ui                = app.debugUi();
    Vulkan::RenderScene* scene = app.scene();
    if (!scene || !ui.section("lighting/GPU and memory", "GPU and memory", false)) return;
    if (DebugUiStyle::beginProperties("##lighting gpu"))
    {
        // The lighting sections against the §17.10 shadow budget, the debug sections' cost separately
        Profiler& p     = app.gpuProfile();
        double lighting = 0, debug = 0;
        for (const char* n : {"environment", "scene.sky", "scene.clouds", "scene.debug", "lighting.preview"})
        {
            const double mean  = p.scopeStats(p.scope(std::string("gpu/") + n), 120).avg;
            const bool isDebug = std::string(n) == "scene.debug" || std::string(n) == "lighting.preview";
            (isDebug ? debug : lighting) += mean;
            DebugUiStyle::property(n);
            ImGui::Text("%.3f ms (120 frame mean)", mean);
        }
        DebugUiStyle::property("lighting / shadow budget");
        ImGui::Text("%.3f of 2.5 ms (§17.10)", lighting);
        DebugUiStyle::property("debug cost");
        ImGui::Text("%.3f ms", debug);
        DebugUiStyle::property("memory");
        ImGui::Text("%.2f MiB", double(scene->lightingStatistics().memoryBytes) / (1 << 20));
        DebugUiStyle::property("preview dispatches");
        ImGui::Text("%u", scene->lightingStatistics().previewDispatches);
        DebugUiStyle::endProperties();
    }
}

}  // namespace Engine
}  // namespace Saiga
