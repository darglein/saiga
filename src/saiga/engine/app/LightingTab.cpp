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
#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>

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
}  // namespace

void LightingTab::draw()
{
    timeAndWeather();
    sunAndMoon();
    exposure();
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
        DebugUiStyle::endProperties();
    }
}

}  // namespace Engine
}  // namespace Saiga
