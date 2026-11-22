/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting's MCP tools: get_lighting, set_time_of_day, set_weather, set_lighting (S36).

#include "saiga/engine/app/LightingTools.h"

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/lighting/EnvironmentExtract.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json       = json11::Json;
using ToolResult = McpServer::ToolResult;

const char* kNoDevice = "this app renders no frames (no device)";

Json schema(const Json::object& properties)
{
    return Json::object{{"type", "object"}, {"properties", properties}, {"additionalProperties", false}};
}
Json arrayProperty(const std::string& description, const Json& items)
{
    return Json::object{{"type", "array"}, {"description", description}, {"items", items}};
}

bool unknownArgument(const Json& args, const std::vector<std::string>& known, std::string& error)
{
    for (auto& kv : args.object_items())
    {
        bool ok = false;
        for (const std::string& k : known) ok = ok || kv.first == k;
        if (!ok)
        {
            std::string list;
            for (const std::string& k : known) list += (list.empty() ? "" : ", ") + k;
            error = "unknown argument \"" + kv.first + "\" (expected: " + list + ")";
            return true;
        }
    }
    return false;
}

std::string rangeText(double lo, double hi)
{
    auto num = [](double v)
    {
        std::string s = std::to_string(v);
        s.erase(s.find_last_not_of('0') + 1);
        if (!s.empty() && s.back() == '.') s.pop_back();
        return s;
    };
    return "[" + num(lo) + ", " + num(hi) + "]";
}

// A finite number in [lo, hi] (the error names the argument and the range)
bool number(const Json& j, const std::string& name, double lo, double hi, double& out, std::string& error)
{
    if (!j.is_number() || !std::isfinite(j.number_value()) || j.number_value() < lo || j.number_value() > hi)
    {
        error = name + ": expected a finite number in " + rangeText(lo, hi);
        return false;
    }
    out = j.number_value();
    return true;
}
bool vec2Arg(const Json& j, const std::string& name, double limit, Vec2& out, std::string& error)
{
    if (!j.is_array() || j.array_items().size() != 2)
    {
        error = name + ": expected [x, z]";
        return false;
    }
    for (int i = 0; i < 2; ++i)
    {
        double v = 0;
        if (!number(j.array_items()[size_t(i)], name + "[" + std::to_string(i) + "]", -limit, limit, v, error))
            return false;
        out(i) = v;
    }
    return true;
}

Json vec3Json(const Vec3& v)
{
    return Json::array{v(0), v(1), v(2)};
}
Json vec3Json(const vec3& v)
{
    return Json::array{double(v(0)), double(v(1)), double(v(2))};
}

const char* toneMapName(Vulkan::ToneMap t)
{
    return t == Vulkan::ToneMap::Aces ? "aces" : "identity";
}
const char* exposureModeName(Vulkan::ExposureMode m)
{
    return m == Vulkan::ExposureMode::Fixed ? "fixed" : "time_of_day";
}
const char* exposureSourceName(Vulkan::ExposureSource s)
{
    switch (s)
    {
        case Vulkan::ExposureSource::Curve:
            return "curve";
        case Vulkan::ExposureSource::Fixed:
            return "fixed";
        default:
            return "enclosed";
    }
}

Json worldTimeJson(const WorldTime& t)
{
    return Json::object{{"hours", t.timeOfDay}, {"day", double(t.day)}, {"rate", t.rate}};
}
Json weatherJson(const Weather& w)
{
    return Json::object{{"cloudiness", w.cloudiness},
                        {"rain", w.rain},
                        {"fog_density", w.fogDensity},
                        {"snow_cover", w.snowCover},
                        {"wind", Json::array{w.wind(0), w.wind(1)}},
                        {"cloud_offset", Json::array{w.cloudOffset(0), w.cloudOffset(1)}},
                        {"wave_amplitude", w.waveAmplitude},
                        {"wave_steepness", w.waveSteepness}};
}

// ---- set_lighting's keys: each reads its value from the settings and writes a validated value into scratch copies

struct LightingState
{
    Vulkan::EnvironmentSettings settings;
    Vulkan::LightingParameters parameters;
    Vulkan::ToneMap toneMap = Vulkan::ToneMap::Identity;
};

struct LightingKey
{
    const char* name;
    bool parameter;  // a LightingParameters field (get_lighting.parameters), else a setting (get_lighting.settings)
    std::function<Json(const LightingState&)> get;
    std::function<bool(const Json&, LightingState&, std::string&)> set;
};

LightingKey floatKey(const char* name, bool parameter, double lo, double hi,
                     std::function<float&(LightingState&)> field)
{
    return LightingKey{name, parameter,
                       [field](const LightingState& s) { return Json(double(field(const_cast<LightingState&>(s)))); },
                       [field, name, lo, hi](const Json& j, LightingState& s, std::string& error)
                       {
                           double v = 0;
                           if (!number(j, name, lo, hi, v, error)) return false;
                           field(s) = float(v);
                           return true;
                       }};
}

LightingKey boolKey(const char* name, bool parameter, std::function<bool&(LightingState&)> field)
{
    return LightingKey{name, parameter,
                       [field](const LightingState& s) { return Json(field(const_cast<LightingState&>(s))); },
                       [field, name](const Json& j, LightingState& s, std::string& error)
                       {
                           if (!j.is_bool())
                           {
                               error = std::string(name) + ": expected true or false";
                               return false;
                           }
                           field(s) = j.bool_value();
                           return true;
                       }};
}

const std::vector<LightingKey>& lightingKeys()
{
    static const std::vector<LightingKey> keys = []
    {
        std::vector<LightingKey> k;
        k.push_back(LightingKey{"tone_map", false, [](const LightingState& s) { return Json(toneMapName(s.toneMap)); },
                                [](const Json& j, LightingState& s, std::string& error)
                                {
                                    if (j == Json("aces"))
                                        s.toneMap = Vulkan::ToneMap::Aces;
                                    else if (j == Json("identity"))
                                        s.toneMap = Vulkan::ToneMap::Identity;
                                    else
                                    {
                                        error = "tone_map: aces or identity";
                                        return false;
                                    }
                                    return true;
                                }});
        k.push_back(LightingKey{"exposure_mode", false,
                                [](const LightingState& s) { return Json(exposureModeName(s.settings.exposure.mode)); },
                                [](const Json& j, LightingState& s, std::string& error)
                                {
                                    if (j == Json("time_of_day"))
                                        s.settings.exposure.mode = Vulkan::ExposureMode::TimeOfDay;
                                    else if (j == Json("fixed"))
                                        s.settings.exposure.mode = Vulkan::ExposureMode::Fixed;
                                    else
                                    {
                                        error = "exposure_mode: time_of_day or fixed";
                                        return false;
                                    }
                                    return true;
                                }});
        k.push_back(
            floatKey("ev100", false, -20, 30, [](LightingState& s) -> float& { return s.settings.exposure.ev100; }));
        k.push_back(floatKey("exposure_compensation", false, -20, 20,
                             [](LightingState& s) -> float& { return s.settings.exposure.compensation; }));
        k.push_back(floatKey("ambient_scale", false, 0, 100,
                             [](LightingState& s) -> float& { return s.settings.ambient.scale; }));
        k.push_back(LightingKey{"min_ambient", false,
                                [](const LightingState& s) { return vec3Json(s.settings.ambient.minAmbient); },
                                [](const Json& j, LightingState& s, std::string& error)
                                {
                                    if (!j.is_array() || j.array_items().size() != 3)
                                    {
                                        error = "min_ambient: expected [r, g, b] (lux)";
                                        return false;
                                    }
                                    vec3 v;
                                    for (int i = 0; i < 3; ++i)
                                    {
                                        double c = 0;
                                        if (!number(j.array_items()[size_t(i)],
                                                    "min_ambient[" + std::to_string(i) + "]", 0, 1000, c, error))
                                            return false;
                                        v(i) = float(c);
                                    }
                                    s.settings.ambient.minAmbient = v;
                                    return true;
                                }});
        k.push_back(
            LightingKey{"sun_override", false,
                        [](const LightingState& s)
                        {
                            if (!s.settings.sunOverride) return Json();
                            return Json(Json::object{{"azimuth_deg", double(s.settings.sunOverride->azimuthDeg)},
                                                     {"elevation_deg", double(s.settings.sunOverride->elevationDeg)}});
                        },
                        [](const Json& j, LightingState& s, std::string& error)
                        {
                            if (j.is_null())
                            {
                                s.settings.sunOverride.reset();
                                return true;
                            }
                            if (!j.is_object())
                            {
                                error = "sun_override: {azimuth_deg, elevation_deg} or null";
                                return false;
                            }
                            for (auto& kv : j.object_items())
                                if (kv.first != "azimuth_deg" && kv.first != "elevation_deg")
                                {
                                    error = "sun_override: unknown key \"" + kv.first + "\"";
                                    return false;
                                }
                            Vulkan::SunOverride o = s.settings.sunOverride.value_or(Vulkan::SunOverride());
                            double v              = 0;
                            if (!j["azimuth_deg"].is_null())
                            {
                                if (!number(j["azimuth_deg"], "sun_override.azimuth_deg", 0, 360, v, error))
                                    return false;
                                o.azimuthDeg = float(v);
                            }
                            if (!j["elevation_deg"].is_null())
                            {
                                if (!number(j["elevation_deg"], "sun_override.elevation_deg", -90, 90, v, error))
                                    return false;
                                o.elevationDeg = float(v);
                            }
                            s.settings.sunOverride = o;
                            return true;
                        }});
        k.push_back(floatKey("cloud_altitude", false, 100, 20000,
                             [](LightingState& s) -> float& { return s.settings.clouds.altitude; }));
        k.push_back(floatKey("cloud_radius", false, 1000, 100000,
                             [](LightingState& s) -> float& { return s.settings.clouds.radius; }));
        k.push_back(floatKey("cloud_shadow_fade", false, 100, 100000,
                             [](LightingState& s) -> float& { return s.settings.clouds.shadowFade; }));
        k.push_back(floatKey("star_intensity", false, 0, 100,
                             [](LightingState& s) -> float& { return s.settings.starIntensity; }));
        k.push_back(floatKey("ground_albedo", false, 0, 1,
                             [](LightingState& s) -> float& { return s.settings.atmosphere.groundAlbedo; }));
        // The LightingParameters (a1)
        k.push_back(boolKey("clouds", true, [](LightingState& s) -> bool& { return s.parameters.clouds; }));
        k.push_back(LightingKey{"environment_resolution", true,
                                [](const LightingState& s) { return Json(s.parameters.environmentResolution); },
                                [](const Json& j, LightingState& s, std::string& error)
                                {
                                    double v = 0;
                                    if (!number(j, "environment_resolution", 32, 512, v, error)) return false;
                                    const int r = int(v);
                                    if (double(r) != v || (r & (r - 1)) != 0)
                                    {
                                        error = "environment_resolution: a power of two in [32, 512]";
                                        return false;
                                    }
                                    s.parameters.environmentResolution = r;
                                    return true;
                                }});
        k.push_back(floatKey("environment_update_degrees", true, 0.01, 90,
                             [](LightingState& s) -> float& { return s.parameters.environmentUpdateDegrees; }));
        k.push_back(boolKey("force_full_updates", true,
                            [](LightingState& s) -> bool& { return s.parameters.forceFullUpdates; }));
        return k;
    }();
    return keys;
}

LightingState currentState(EngineApp& app)
{
    LightingState s;
    s.settings = app.environmentSettings();
    if (Vulkan::RenderScene* scene = app.scene())
    {
        s.parameters = scene->lightingParameters();
        s.toneMap    = scene->parameters().toneMap;
    }
    return s;
}

std::vector<std::string> keyNames()
{
    std::vector<std::string> names;
    for (const LightingKey& k : lightingKeys()) names.push_back(k.name);
    return names;
}

Json keysJson(const LightingState& s, bool parameters)
{
    Json::object o;
    for (const LightingKey& k : lightingKeys())
        if (k.parameter == parameters && std::string(k.name) != "tone_map") o[k.name] = k.get(s);
    return o;
}

const std::vector<std::string>& sectionNames()
{
    static const std::vector<std::string> names = {"time",         "weather",  "sun",       "moon",
                                                   "shadow_light", "tone_map", "exposure",  "ambient",
                                                   "clouds",       "settings", "parameters"};
    return names;
}
}  // namespace

World& lightingWriteWorld(EngineApp& app)
{
    return app.coop() ? *app.hostWorld() : app.world();
}

bool lightingJson(EngineApp& app, const std::vector<std::string>& sections, Json& out, std::string& error)
{
    for (const std::string& s : sections)
        if (std::find(sectionNames().begin(), sectionNames().end(), s) == sectionNames().end())
        {
            std::string list;
            for (const std::string& n : sectionNames()) list += (list.empty() ? "" : ", ") + n;
            error = "sections: unknown section \"" + s + "\" (sections: " + list + ")";
            return false;
        }
    Vulkan::RenderScene* scene = app.scene();
    if (!scene)
    {
        error = kNoDevice;
        return false;
    }
    const EnvironmentSample e  = app.sampleViewEnvironment();
    const LightingState state  = currentState(app);
    const Vulkan::ToneMap tone = state.toneMap;
    Json::object o;
    o["time"] =
        e.hasTime ? Json(Json::object{{"hours", e.time.timeOfDay}, {"day", double(e.time.day)}, {"rate", e.time.rate}})
                  : Json();
    o["weather"]            = Json(Json::object{
                   {"space", int(app.viewCamera().space)}, {"enclosed", e.enclosed}, {"values", weatherJson(e.weather)}});
    const CelestialState& c = e.celestial;
    o["sun"] =
        Json::object{{"direction", vec3Json(c.sunDir)},
                     {"azimuth_deg", c.sunAzimuthDeg},
                     {"elevation_deg", c.sunElevationDeg},
                     {"illuminance_top", double(e.state.sunIlluminance)},
                     {"illuminance_ground", groundIlluminance(double(e.state.sunIlluminance), c.sunElevationDeg)},
                     {"override", state.settings.sunOverride.has_value()}};
    o["moon"] =
        Json::object{{"direction", vec3Json(c.moonDir)},
                     {"azimuth_deg", c.moonAzimuthDeg},
                     {"elevation_deg", c.moonElevationDeg},
                     {"illuminance_top", double(e.state.moonIlluminance)},
                     {"illuminance_ground", groundIlluminance(double(e.state.moonIlluminance), c.moonElevationDeg)},
                     {"phase", c.moonPhase}};
    o["shadow_light"] = shadowLightName(e.shadowLight);
    if (tone == Vulkan::ToneMap::Identity)
        o["tone_map"] = Json::object{
            {"value", toneMapName(tone)},
            {"note", "identity is the clay view: it lights with the studio light and ignores time and weather"}};
    else
        o["tone_map"] = Json::object{{"value", toneMapName(tone)}};
    if (scene->hasEnvironment())
        o["exposure"] = Json::object{{"mode", exposureModeName(state.settings.exposure.mode)},
                                     {"source", exposureSourceName(scene->environment().exposureSource)},
                                     {"ev100", double(scene->environment().ev100)},
                                     {"curve_ev100", double(e.curveEv100)},
                                     {"compensation", double(state.settings.exposure.compensation)},
                                     {"multiplier", double(scene->exposure())},
                                     {"effective", double(scene->frameExposure())},
                                     {"pre_exposure", double(scene->preExposure())}};
    else
        o["exposure"] = Json::object{{"environment", false},
                                     {"multiplier", double(scene->exposure())},
                                     {"effective", double(scene->frameExposure())},
                                     {"note", "no environment: the unit-less exposure multiplier"}};
    o["ambient"]    = Json::object{{"scale", double(state.settings.ambient.scale)},
                                   {"min", vec3Json(state.settings.ambient.minAmbient)},
                                   {"enclosed", e.enclosed},
                                   {"enclosed_ambient", vec3Json(e.state.enclosedAmbient)}};
    o["clouds"]     = Json::object{{"altitude", double(state.settings.clouds.altitude)},
                                   {"radius", double(state.settings.clouds.radius)},
                                   {"seed", double(state.settings.clouds.seed)},
                                   {"shadow_fade", double(state.settings.clouds.shadowFade)},
                                   {"fog_falloff", double(state.settings.clouds.fogFalloff)},
                                   {"enabled", state.parameters.clouds}};
    o["settings"]   = keysJson(state, false);
    o["parameters"] = keysJson(state, true);
    if (!sections.empty())
    {
        Json::object filtered;
        for (const std::string& s : sections) filtered[s] = o[s];
        o = filtered;
    }
    out = o;
    return true;
}

Json lightingStatusJson(EngineApp& app)
{
    Vulkan::RenderScene* scene = app.scene();
    if (!scene) return Json();
    const World& w = app.world();
    return Json::object{{"time_of_day", hasWorldTime(w) ? Json(app.sampleViewEnvironment().time.timeOfDay) : Json()},
                        {"ev100", scene->hasEnvironment() ? Json(double(scene->environment().ev100)) : Json()},
                        {"tone_map", toneMapName(scene->parameters().toneMap)},
                        {"debug_view", "none"}};
}

void registerLightingTools(McpServer& server, EngineApp& app)
{
    server.addTool(
        "get_lighting",
        "Reads the lighting of the view camera's space: time (hours, day, rate; null without WorldTime), weather, sun "
        "and moon (direction, azimuth and elevation in degrees, illuminance at the top of the atmosphere and on the "
        "ground, the moon's phase), shadow_light, tone_map (identity = clay, which ignores time and weather), "
        "exposure (mode, source, ev100, curve_ev100, compensation, multiplier, effective, pre_exposure; without an "
        "environment only the unit-less multiplier), ambient, clouds, settings (the set_lighting keys) and parameters "
        "(the LightingParameters). sections keeps only the named top-level keys. Answered at once. Changes nothing.",
        schema({{"sections", arrayProperty("The top-level keys to return (default all)",
                                           McpServer::StringProperty("", sectionNames()))}}),
        [&app](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a, {"sections"}, error)) return respond(ToolResult::Error(error));
            std::vector<std::string> sections;
            if (!a["sections"].is_null())
            {
                if (!a["sections"].is_array()) return respond(ToolResult::Error("sections: a list of section names"));
                for (const Json& s : a["sections"].array_items())
                {
                    if (!s.is_string()) return respond(ToolResult::Error("sections: a list of section names"));
                    sections.push_back(s.string_value());
                }
            }
            Json out;
            if (!lightingJson(app, sections, out, error)) return respond(ToolResult::Error(error));
            respond(ToolResult::Text(out.dump()));
        });

    server.addTool(
        "set_time_of_day",
        "Changes the world's WorldTime: hours [0, 24), day (integer), rate (game seconds per simulated second, "
        "[-3600, 3600]; 0 freezes the time, 60 is a game hour per minute). Returns values and previous. An error "
        "without WorldTime (EnvironmentSetup::enabled); with --coop it changes the hidden host's WorldTime (the "
        "client gets it by replication), a pure client is refused. Changes world state (replicated, saved).",
        schema({{"hours", McpServer::NumberProperty("Time of day in hours, [0, 24)")},
                {"day", McpServer::IntegerProperty("The day count")},
                {"rate", McpServer::NumberProperty("Game seconds per simulated second, [-3600, 3600]")}}),
        [&app](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a, {"hours", "day", "rate"}, error)) return respond(ToolResult::Error(error));
            WorldTimeChange c;
            double v = 0;
            if (!a["hours"].is_null())
            {
                if (!number(a["hours"], "hours", 0, 24, v, error) || v >= 24)
                    return respond(ToolResult::Error("hours: expected a finite number in [0, 24)"));
                c.hours = v;
            }
            if (!a["day"].is_null())
            {
                if (!number(a["day"], "day", -1e15, 1e15, v, error) || v != std::floor(v))
                    return respond(ToolResult::Error("day: expected an integer"));
                c.day = int64_t(v);
            }
            if (!a["rate"].is_null())
            {
                if (!number(a["rate"], "rate", -kMaxTimeRate, kMaxTimeRate, v, error))
                    return respond(ToolResult::Error(error));
                c.rate = v;
            }
            World& w = lightingWriteWorld(app);
            WorldTime previous;
            if (!setWorldTime(w, c, error, &previous)) return respond(ToolResult::Error(error));
            Json::object o{{"values", worldTimeJson(w.readGlobal<WorldTime>())}, {"previous", worldTimeJson(previous)}};
            if (app.coop()) o["world"] = "host";
            respond(ToolResult::Text(Json(o).dump()));
        });

    server.addTool(
        "set_weather",
        "Changes the Weather of a space (default: the view camera's space): cloudiness, rain, snow_cover [0, 1], "
        "fog_density [0, 0.05] (1/m), wind [x, z] (m/s), cloud_offset [x, z] (m, wrapped to the cloud tile), "
        "wave_amplitude [0, 100], wave_steepness [0, 1]. Returns values and previous. An error without Weather "
        "(EnvironmentSetup::enabled); with --coop it changes the hidden host's, a pure client is refused. Changes "
        "world state (replicated, saved).",
        schema({{"space", McpServer::IntegerProperty("The space (default: the view camera's)")},
                {"cloudiness", McpServer::NumberProperty("[0, 1]")},
                {"rain", McpServer::NumberProperty("[0, 1] (no shading effect before S43)")},
                {"fog_density", McpServer::NumberProperty("[0, 0.05] per metre")},
                {"snow_cover", McpServer::NumberProperty("[0, 1] (no shading effect before S43)")},
                {"wind", arrayProperty("[x, z] in m/s", Json::object{{"type", "number"}})},
                {"cloud_offset", arrayProperty("[x, z] in m", Json::object{{"type", "number"}})},
                {"wave_amplitude", McpServer::NumberProperty("[0, 100]")},
                {"wave_steepness", McpServer::NumberProperty("[0, 1]")}}),
        [&app](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a,
                                {"space", "cloudiness", "rain", "fog_density", "snow_cover", "wind", "cloud_offset",
                                 "wave_amplitude", "wave_steepness"},
                                error))
                return respond(ToolResult::Error(error));
            World& w      = lightingWriteWorld(app);
            SpaceId space = app.viewCamera().space;
            double v      = 0;
            if (!a["space"].is_null())
            {
                if (!number(a["space"], "space", 0, 65535, v, error) || v != std::floor(v))
                    return respond(ToolResult::Error("space: expected a space index"));
                space = SpaceId(v);
            }
            WeatherChange c;
            struct Field
            {
                const char* name;
                double lo, hi;
                std::optional<double>* target;
            };
            const Field fields[] = {{"cloudiness", 0, 1, &c.cloudiness},
                                    {"rain", 0, 1, &c.rain},
                                    {"fog_density", 0, kMaxFogDensity, &c.fogDensity},
                                    {"snow_cover", 0, 1, &c.snowCover},
                                    {"wave_amplitude", 0, kMaxWaveAmplitude, &c.waveAmplitude},
                                    {"wave_steepness", 0, 1, &c.waveSteepness}};
            for (const Field& f : fields)
                if (!a[f.name].is_null())
                {
                    if (!number(a[f.name], f.name, f.lo, f.hi, v, error)) return respond(ToolResult::Error(error));
                    *f.target = v;
                }
            Vec2 xz = Vec2::Zero();
            if (!a["wind"].is_null())
            {
                if (!vec2Arg(a["wind"], "wind", kMaxWind, xz, error)) return respond(ToolResult::Error(error));
                c.wind = xz;
            }
            if (!a["cloud_offset"].is_null())
            {
                if (!vec2Arg(a["cloud_offset"], "cloud_offset", 1e12, xz, error))
                    return respond(ToolResult::Error(error));
                c.cloudOffset = xz;
            }
            Weather previous;
            if (!setWeather(w, space, c, error, &previous)) return respond(ToolResult::Error(error));
            Json::object o{{"space", int(space)},
                           {"values", weatherJson(w.readSpaceState<Weather>(space))},
                           {"previous", weatherJson(previous)}};
            if (app.coop()) o["world"] = "host";
            respond(ToolResult::Text(Json(o).dump()));
        });

    Json::object setProperties;
    for (const LightingKey& k : lightingKeys())
    {
        const std::string n = k.name;
        Json p;
        if (n == "tone_map")
            p = McpServer::StringProperty("aces, or identity (the clay view: studio light)", {"aces", "identity"});
        else if (n == "exposure_mode")
            p = McpServer::StringProperty("time_of_day (the curve by sun elevation) or fixed (ev100)",
                                          {"time_of_day", "fixed"});
        else if (n == "min_ambient")
            p = arrayProperty("[r, g, b] lux", Json::object{{"type", "number"}});
        else if (n == "sun_override")
            p = Json::object{{"description", "{azimuth_deg, elevation_deg} or null"}};
        else if (n == "clouds" || n == "force_full_updates")
            p = McpServer::BooleanProperty(n);
        else if (n == "environment_resolution")
            p = McpServer::IntegerProperty("A power of two in [32, 512]");
        else
            p = McpServer::NumberProperty(n);
        setProperties[n] = p;
    }
    server.addTool(
        "set_lighting",
        "Changes the lighting settings (not world state: not replicated, not saved), validated as a whole and applied "
        "all or nothing from the next frame: tone_map (aces, identity), exposure_mode (time_of_day, fixed), ev100, "
        "exposure_compensation (EV, +1 is twice as bright), ambient_scale, min_ambient ([r, g, b] lux), sun_override "
        "({azimuth_deg, elevation_deg} or null), cloud_altitude, cloud_radius, cloud_shadow_fade (m), star_intensity, "
        "ground_albedo, and the LightingParameters clouds, environment_resolution, environment_update_degrees, "
        "force_full_updates. Returns values and previous (previous is an undo payload of this tool). An error names "
        "every invalid key and changes nothing.",
        schema(setProperties),
        [&app](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a, keyNames(), error)) return respond(ToolResult::Error(error));
            if (!app.scene()) return respond(ToolResult::Error(kNoDevice));
            const LightingState before = currentState(app);
            LightingState after        = before;
            std::string errors;
            Json::object values, previous;
            for (const LightingKey& k : lightingKeys())
            {
                const Json& v = a[k.name];
                // A key that is present: null only for sun_override
                const bool present = a.object_items().count(k.name) != 0;
                if (!present) continue;
                std::string e;
                if (!k.set(v, after, e))
                {
                    errors += (errors.empty() ? "" : "; ") + e;
                    continue;
                }
                previous[k.name] = k.get(before);
            }
            if (!errors.empty()) return respond(ToolResult::Error(errors + " (nothing changed)"));
            for (auto& kv : previous)
                values[kv.first] = [&]
                {
                    for (const LightingKey& k : lightingKeys())
                        if (kv.first == k.name) return k.get(after);
                    return Json();
                }();
            app.setEnvironmentSettings(after.settings);
            app.scene()->setLightingParameters(after.parameters);
            app.scene()->setToneMap(after.toneMap);
            respond(ToolResult::Text(Json(Json::object{{"values", values}, {"previous", previous}}).dump()));
        });
}

}  // namespace Engine
}  // namespace Saiga
