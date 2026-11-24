/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The lighting's MCP tools: get_lighting, set_time_of_day, set_weather, set_lighting (S36a1a), set_lighting_debug,
// capture_lighting, probe_lighting (S36a1b).

#include "saiga/engine/app/LightingTools.h"

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/app/EngineApp.h"
#include "saiga/engine/lighting/EnvironmentExtract.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/tools/WorldTools.h"
#include "saiga/vulkan/scene/EnvironmentMap.h"
#include "saiga/vulkan/scene/LightClusters.h"
#include "saiga/vulkan/scene/LightingDebug.h"
#include "saiga/vulkan/scene/LocalLights.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/scene/ShadowAtlas.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json       = json11::Json;
using ToolResult = McpServer::ToolResult;

const char* kNoDevice = "this app renders no frames (no device)";

// capture_lighting's image names: every LightingImage but the cascade strip ("cascade" without its argument;
// round 1, finding 6: the schema's list was written by hand and missed atlas and atlas_tile)
std::vector<std::string> captureImageNames()
{
    std::vector<std::string> names;
    for (int i = 0; i < int(Vulkan::LightingImage::Count); ++i)
        if (Vulkan::LightingImage(i) != Vulkan::LightingImage::CascadeStrip)
            names.push_back(Vulkan::lightingImageName(Vulkan::LightingImage(i)));
    return names;
}

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
    // The schema's description (the range of a number key)
    std::string description = {};
};

LightingKey floatKey(const char* name, bool parameter, double lo, double hi,
                     std::function<float&(LightingState&)> field)
{
    return LightingKey{name, parameter,
                       [field](const LightingState& s) { return Json(double(field(const_cast<LightingState&>(s)))); },
                       [field, name, lo, hi](const Json& j, LightingState& s, std::string& error)
                       {
                           // The stored float against the bounds as floats (round 2, finding 17: 0.01 is stored as
                           // 0.0099999998, which failed the >= 0.01 check when it was sent back, e.g. the undo
                           // payload or a saved environment file)
                           const float f = j.is_number() ? float(j.number_value()) : 0.0f;
                           if (!j.is_number() || !std::isfinite(f) || !(f >= float(lo) && f <= float(hi)))
                           {
                               error = std::string(name) + ": expected a finite number in " + rangeText(lo, hi);
                               return false;
                           }
                           field(s) = f;
                           return true;
                       },
                       std::string(name) + " " + rangeText(lo, hi)};
}

LightingKey intKey(const char* name, bool parameter, int lo, int hi, std::function<int&(LightingState&)> field,
                   bool powerOfTwo = false)
{
    return LightingKey{name, parameter,
                       [field](const LightingState& s) { return Json(field(const_cast<LightingState&>(s))); },
                       [field, name, lo, hi, powerOfTwo](const Json& j, LightingState& s, std::string& error)
                       {
                           double v = 0;
                           if (!number(j, name, lo, hi, v, error)) return false;
                           const int i = int(v);
                           if (double(i) != v || (powerOfTwo && (i & (i - 1)) != 0))
                           {
                               error = std::string(name) + ": " + (powerOfTwo ? "a power of two" : "an integer") +
                                       " in " + rangeText(lo, hi);
                               return false;
                           }
                           field(s) = i;
                           return true;
                       },
                       // (round 2, finding 15: the schema showed the integer keys with their names only)
                       std::string(name) + " " + rangeText(lo, hi) + (powerOfTwo ? ", a power of two" : "")};
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
        // The a1c review: the rest of what the tab edits (the atmosphere's scales, the fog falloff, the cloud seed)
        k.push_back(floatKey("rayleigh_scale", false, 0, 10,
                             [](LightingState& s) -> float& { return s.settings.atmosphere.rayleighScale; }));
        k.push_back(floatKey("mie_scale", false, 0, 10,
                             [](LightingState& s) -> float& { return s.settings.atmosphere.mieScale; }));
        k.push_back(floatKey("ozone_scale", false, 0, 10,
                             [](LightingState& s) -> float& { return s.settings.atmosphere.ozoneScale; }));
        k.push_back(floatKey("fog_falloff", false, 0, 0.1,
                             [](LightingState& s) -> float& { return s.settings.clouds.fogFalloff; }));
        k.push_back(LightingKey{"cloud_seed", false,
                                [](const LightingState& s) { return Json(double(s.settings.clouds.seed)); },
                                [](const Json& j, LightingState& s, std::string& error)
                                {
                                    double v = 0;
                                    if (!number(j, "cloud_seed", 0, double(Vulkan::kMaxCloudSeed), v, error))
                                        return false;
                                    if (v != std::floor(v))
                                    {
                                        error = "cloud_seed: an integer";
                                        return false;
                                    }
                                    s.settings.clouds.seed = uint32_t(v);
                                    return true;
                                },
                                "cloud_seed [0, 2147483647] (an integer)"});
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
        // S36a2: the cascades
        using P = Vulkan::LightingParameters;
        auto pf = [](float P::* f)
        {
            return [f](LightingState& s) -> float& { return s.parameters.*f; };
        };
        auto pi = [](int P::* f)
        {
            return [f](LightingState& s) -> int& { return s.parameters.*f; };
        };
        auto pb = [](bool P::* f)
        {
            return [f](LightingState& s) -> bool& { return s.parameters.*f; };
        };
        k.push_back(boolKey("sun_shadows", true, pb(&P::sunShadows)));
        k.push_back(intKey("cascade_count", true, 1, 4, pi(&P::cascadeCount)));
        k.push_back(intKey("cascade_resolution", true, 256, 4096, pi(&P::cascadeResolution), true));
        k.push_back(floatKey("shadow_distance", true, 10, 20000, pf(&P::shadowDistance)));
        k.push_back(floatKey("split_lambda", true, 0, 1, pf(&P::splitLambda)));
        k.push_back(LightingKey{
            "cascade_interval", true,
            [](const LightingState& s)
            {
                Json::array a;
                for (int v : s.parameters.cascadeInterval) a.push_back(v);
                return Json(a);
            },
            [](const Json& j, LightingState& s, std::string& error)
            {
                if (!j.is_array() || j.array_items().size() != 4)
                {
                    error = "cascade_interval: [4] integers in [1, 8], the first 1";
                    return false;
                }
                std::array<int, 4> v;
                for (size_t i = 0; i < 4; ++i)
                {
                    double d = 0;
                    if (!number(j[i], "cascade_interval[" + std::to_string(i) + "]", 1, 8, d, error)) return false;
                    if (d != std::floor(d))
                    {
                        error = "cascade_interval[" + std::to_string(i) + "]: an integer in [1, 8]";
                        return false;
                    }
                    v[i] = int(d);
                }
                if (v[0] != 1)
                {
                    error = "cascade_interval[0]: must be 1 (cascade 0 is fitted to the camera frustum every frame)";
                    return false;
                }
                s.parameters.cascadeInterval = v;
                return true;
            }});
        k.push_back(floatKey("depth_bias_texels", true, 0, 20, pf(&P::depthBiasTexels)));
        k.push_back(floatKey("slope_bias", true, 0, 20, pf(&P::slopeBias)));
        k.push_back(floatKey("normal_offset_texels", true, 0, 20, pf(&P::normalOffsetTexels)));
        k.push_back(floatKey("cascade_blend", true, 0, 0.5, pf(&P::cascadeBlend)));
        k.push_back(intKey("shadow_lod_bias", true, 0, 3, pi(&P::shadowLodBias)));
        k.push_back(floatKey("small_caster_texels", true, 0, 100, pf(&P::smallCasterTexels)));
        k.push_back(floatKey("max_caster_height", true, 0, 10000, pf(&P::maxCasterHeight)));
        k.push_back(floatKey("min_shadow_elevation_deg", true, 0.5, 45, pf(&P::minShadowElevationDeg)));
        k.push_back(floatKey("cascade_caster_distance", true, 0, 20000, pf(&P::cascadeCasterDistance)));
        k.push_back(boolKey("terrain_casts_in_cascades", true, pb(&P::terrainCastsInCascades)));
        // S36a2: the horizon map
        k.push_back(boolKey("horizon_shadows", true, pb(&P::horizonShadows)));
        k.push_back(intKey("horizon_resolution", true, 128, 4096, pi(&P::horizonResolution), true));
        k.push_back(floatKey("horizon_extent", true, 1000, 200000, pf(&P::horizonExtent)));
        k.push_back(intKey("horizon_rows_per_frame", true, 1, 4096, pi(&P::horizonRowsPerFrame)));
        // S36b: the local lights
        k.push_back(boolKey("local_lights", true, pb(&P::localLights)));
        k.push_back(boolKey("clustered_lights", true, pb(&P::clusteredLights)));
        k.push_back(intKey("max_lights", true, 1, 4096, pi(&P::maxLights)));
        k.push_back(intKey("cluster_x", true, 1, 64, pi(&P::clusterX)));
        k.push_back(intKey("cluster_y", true, 1, 64, pi(&P::clusterY)));
        k.push_back(intKey("cluster_z", true, 2, 128, pi(&P::clusterZ)));
        k.push_back(floatKey("cluster_near", true, 0.01, 1000, pf(&P::clusterNear)));
        k.push_back(floatKey("cluster_far", true, 1, 100000, pf(&P::clusterFar)));
        k.push_back(intKey("max_lights_per_cluster", true, 1, 1024, pi(&P::maxLightsPerCluster)));
        k.push_back(intKey("atlas_resolution", true, 512, 8192, pi(&P::atlasResolution), true));
        k.push_back(intKey("atlas_tile", true, 64, 8192, pi(&P::atlasTile)));
        k.push_back(intKey("faces_per_frame", true, 0, 11, pi(&P::facesPerFrame)));
        k.push_back(floatKey("atlas_depth_bias", true, 0, 1000, pf(&P::atlasDepthBias)));
        k.push_back(floatKey("atlas_normal_offset", true, 0, 20, pf(&P::atlasNormalOffset)));
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
    static const std::vector<std::string> names = {"time",     "weather",    "sun",     "moon",        "shadow_light",
                                                   "tone_map", "exposure",   "ambient", "environment", "clouds",
                                                   "settings", "parameters", "debug",   "gpu",         "memory",
                                                   "cascades", "horizon",    "lights",  "clusters",    "atlas"};
    return names;
}
}  // namespace

bool validateLightingValues(const Vulkan::EnvironmentSettings& settings, const Vulkan::LightingParameters& parameters,
                            std::string& error)
{
    // Every set_lighting key reads its value and validates it as the tool would (the same ranges)
    LightingState state;
    state.settings   = settings;
    state.parameters = parameters;
    std::string errors;
    for (const LightingKey& k : lightingKeys())
    {
        LightingState scratch = state;
        std::string e;
        if (!k.set(k.get(state), scratch, e)) errors += (errors.empty() ? "" : "; ") + e;
    }
    // The cross-key rules (finding 23 of round 1; round 2, finding 9: also for the tab and the environment file)
    if (parameters.clusterNear >= parameters.clusterFar)
        errors += std::string(errors.empty() ? "" : "; ") + "cluster_near (" + std::to_string(parameters.clusterNear) +
                  ") must be below cluster_far (" + std::to_string(parameters.clusterFar) + ")";
    if (parameters.atlasTile > parameters.atlasResolution ||
        parameters.atlasResolution % std::max(parameters.atlasTile, 1) != 0)
        errors += std::string(errors.empty() ? "" : "; ") + "atlas_tile (" + std::to_string(parameters.atlasTile) +
                  ") must divide atlas_resolution (" + std::to_string(parameters.atlasResolution) + ")";
    const std::vector<vec2>& curve = settings.exposure.curve;
    if (curve.empty()) errors += std::string(errors.empty() ? "" : "; ") + "exposure curve: empty";
    for (size_t i = 0; i < curve.size(); ++i)
    {
        if (!std::isfinite(curve[i](0)) || !std::isfinite(curve[i](1)))
            errors += std::string(errors.empty() ? "" : "; ") + "exposure curve[" + std::to_string(i) + "]: not finite";
        else if (i > 0 && !(curve[i](0) > curve[i - 1](0)))
            errors += std::string(errors.empty() ? "" : "; ") + "exposure curve[" + std::to_string(i) +
                      "]: the elevations must be strictly ascending";
    }
    if (errors.empty()) return true;
    error = errors;
    return false;
}

LightingMemoryEstimate lightingMemoryAfter(const Vulkan::LightingStatistics& ls, const Vulkan::LightingParameters& p)
{
    // The cascades (D32 layers) and the horizon maps (three RG32F maps) while on
    const uint64_t cascadeBytes =
        p.sunShadows ? uint64_t(p.cascadeResolution) * uint64_t(p.cascadeResolution) * 4 * uint64_t(p.cascadeCount) : 0;
    const uint64_t horizonBytes =
        p.horizonShadows ? uint64_t(p.horizonResolution) * uint64_t(p.horizonResolution) * 3 * 8 : 0;
    // S36b: the atlas (two D16 layers) while the local lights are on
    const uint64_t atlasBytes = p.localLights ? uint64_t(p.atlasResolution) * uint64_t(p.atlasResolution) * 4 : 0;
    const uint64_t atlasNow   = uint64_t(ls.atlas.resolution) * uint64_t(ls.atlas.resolution) * 4;
    // The cluster buffers (LightClusters::ensure; round 2, finding 2): per frame slot the (count, total) table and the
    // light indices, the shared bounds (the per-light counts are small)
    const uint64_t clusters =
        uint64_t(std::max(p.clusterX, 0)) * uint64_t(std::max(p.clusterY, 0)) * uint64_t(std::max(p.clusterZ, 0));
    const uint64_t indicesPerSlot = clusters * uint64_t(std::max(p.maxLightsPerCluster, 0)) * 4;
    const uint64_t clusterBytes =
        p.localLights && p.clusteredLights
            ? (clusters * 8 + indicesPerSlot) * Vulkan::VulkanRenderer::framesInFlight + clusters * 32
            : 0;
    LightingMemoryEstimate m;
    // (the horizon maps are replaced by the estimate, the resident chain stays: CodeRabbit #85 round 2)
    const uint64_t replaced = ls.shadowMemoryBytes + (ls.horizon.memoryBytes - ls.horizon.chainBytes) + atlasNow +
                              ls.lights.clusterMemory;
    m.total = (ls.memoryBytes > replaced ? ls.memoryBytes - replaced : 0) + cascadeBytes + horizonBytes + atlasBytes +
              clusterBytes;
    m.largestBinding = p.localLights && p.clusteredLights ? indicesPerSlot : 0;
    return m;
}

bool validateLighting(const Vulkan::LightingStatistics& now, uint64_t storageLimit,
                      const Vulkan::EnvironmentSettings& settings, const Vulkan::LightingParameters& parameters,
                      std::string& error, uint64_t* memory)
{
    if (!validateLightingValues(settings, parameters, error)) return false;
    const LightingMemoryEstimate m = lightingMemoryAfter(now, parameters);
    if (memory) *memory = m.total;
    if (m.total > kLightingMemoryLimit)
    {
        error = "the lighting memory after the change (" + std::to_string(m.total) +
                " bytes) exceeds 1 GiB (cascade_resolution, cascade_count, horizon_resolution, atlas_resolution, "
                "cluster_x * cluster_y * cluster_z * max_lights_per_cluster)";
        return false;
    }
    if (m.largestBinding > storageLimit)
    {
        error = "the cluster light indices (cluster_x * cluster_y * cluster_z * max_lights_per_cluster * 4 = " +
                std::to_string(m.largestBinding) + " bytes) exceed the device's storage buffer limit (" +
                std::to_string(storageLimit) + " bytes)";
        return false;
    }
    return true;
}

namespace
{
// The lighting's GPU sections of S36a1b (a1c, a2 and b add theirs) and the §17.10 shadow budget
const std::vector<std::string>& lightingGpuSections()
{
    static const std::vector<std::string> names = {"environment",    "lights",          "shadow",    "shadow.cascades",
                                                   "shadow.horizon", "shadow.atlas",    "scene.sky", "scene.clouds",
                                                   "scene.debug",    "lighting.preview"};
    return names;
}

bool lightingDebugGpuSection(const std::string& name)
{
    return name == "scene.debug" || name == "lighting.preview";
}

Json mappingJson(const Vulkan::PreviewMapping& m)
{
    return Json::object{{"range", m.autoRange ? Json("auto") : Json(Json::array{double(m.min), double(m.max)})},
                        {"log", m.log},
                        {"color_map", m.colorMap == Vulkan::ColorMap::Turbo ? "turbo" : "gray"}};
}

// The mapping arguments of capture_lighting / set_lighting_debug: range ("auto" or [min, max]), log, color_map
bool parseMapping(const Json& range, const Json& log, const Json& colorMap, Vulkan::PreviewMapping& m,
                  std::string& error)
{
    if (!range.is_null())
    {
        if (range == Json("auto"))
            m.autoRange = true;
        else if (range.is_array() && range.array_items().size() == 2 && range[0].is_number() && range[1].is_number() &&
                 std::isfinite(range[0].number_value()) && std::isfinite(range[1].number_value()) &&
                 range[0].number_value() < range[1].number_value())
        {
            m.autoRange = false;
            m.min       = float(range[0].number_value());
            m.max       = float(range[1].number_value());
        }
        else
        {
            error = "range: \"auto\" or [min, max] with min < max";
            return false;
        }
    }
    if (!log.is_null())
    {
        if (!log.is_bool())
        {
            error = "log: true or false";
            return false;
        }
        m.log = log.bool_value();
    }
    if (!colorMap.is_null())
    {
        if (colorMap == Json("gray"))
            m.colorMap = Vulkan::ColorMap::Gray;
        else if (colorMap == Json("turbo"))
            m.colorMap = Vulkan::ColorMap::Turbo;
        else
        {
            error = "color_map: gray or turbo";
            return false;
        }
    }
    return true;
}

std::vector<std::string> viewNames()
{
    std::vector<std::string> names;
    for (int i = 0; i < int(Vulkan::LightingDebugView::Count); ++i)
        names.push_back(Vulkan::lightingDebugViewName(Vulkan::LightingDebugView(i)));
    return names;
}
std::vector<std::string> overlayNames()
{
    std::vector<std::string> names;
    for (const auto& o : Vulkan::lightingOverlayNames()) names.push_back(o.first);
    return names;
}
std::string joined(const std::vector<std::string>& v)
{
    std::string r;
    for (const std::string& s : v) r += (r.empty() ? "" : ", ") + s;
    return r;
}
}  // namespace

bool parseLightingDebug(const Json& view, const Json& overlays, Vulkan::LightingDebugState& out, std::string& error)
{
    out = Vulkan::LightingDebugState();
    if (!view.is_null())
    {
        const auto v = view.is_string() ? Vulkan::lightingDebugViewFromName(view.string_value()) : std::nullopt;
        if (!v)
        {
            error = "lighting_view: unknown view " + view.dump() + " (views: " + joined(viewNames()) + ")";
            return false;
        }
        out.view = *v;
    }
    if (!overlays.is_null())
    {
        if (!overlays.is_array())
        {
            error = "overlays: a list of overlay names (" + joined(overlayNames()) + ")";
            return false;
        }
        for (const Json& o : overlays.array_items())
        {
            bool found = false;
            for (const auto& n : Vulkan::lightingOverlayNames())
                if (o.is_string() && o.string_value() == n.first)
                {
                    out.overlays |= n.second;
                    found = true;
                }
            if (!found)
            {
                error = "overlays: unknown overlay " + o.dump() + " (overlays: " + joined(overlayNames()) + ")";
                return false;
            }
        }
    }
    return true;
}

Json lightingDebugJson(const Vulkan::LightingDebugState& state)
{
    Json::array overlays;
    for (const auto& n : Vulkan::lightingOverlayNames())
        if (state.overlays & n.second) overlays.push_back(n.first);
    return Json::object{{"view", Vulkan::lightingDebugViewName(state.view)},
                        {"overlays", overlays},
                        {"highlight_light", state.highlightLight ? Json(double(state.highlightLight)) : Json()}};
}

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
    {
        // ev100, source and effective are the sample's (now, with the current settings: what the next frame uses), so
        // they agree with curve_ev100 and compensation right after a set_lighting; frame_ev100 and frame_effective are
        // the last rendered frame's
        const Vulkan::RenderSceneParameters& p = scene->parameters();
        const float multiplier                 = p.toneMap == Vulkan::ToneMap::Identity ? 1.0f : p.exposure;
        const float effective =
            p.toneMap == Vulkan::ToneMap::Identity ? 1.0f : Vulkan::exposureFromEv100(e.state.ev100, p.exposure);
        o["exposure"] = Json::object{{"mode", exposureModeName(state.settings.exposure.mode)},
                                     {"source", exposureSourceName(e.state.exposureSource)},
                                     {"ev100", double(e.state.ev100)},
                                     {"curve_ev100", double(e.curveEv100)},
                                     {"compensation", double(state.settings.exposure.compensation)},
                                     {"multiplier", double(multiplier)},
                                     {"effective", double(effective)},
                                     {"frame_ev100", double(scene->environment().ev100)},
                                     {"frame_effective", double(scene->frameExposure())},
                                     {"pre_exposure", double(scene->preExposure())}};
    }
    else
        o["exposure"] = Json::object{{"environment", false},
                                     {"multiplier", double(scene->exposure())},
                                     {"effective", double(scene->frameExposure())},
                                     {"note", "no environment: the unit-less exposure multiplier"}};
    const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
    Json::array sh;
    for (const vec3& c : ls.sh) sh.push_back(vec3Json(c));
    o["ambient"] = Json::object{{"scale", double(state.settings.ambient.scale)},
                                {"min", vec3Json(state.settings.ambient.minAmbient)},
                                {"enclosed", e.enclosed},
                                {"enclosed_ambient", vec3Json(e.state.enclosedAmbient)},
                                {"sh9", sh},
                                {"dominant", vec3Json(ls.sh[0])}};
    // S36a1c: the environment map (the frame of its last complete capture, the progress of the one in progress) and the
    // atmosphere's LUT recomputations
    {
        const Vulkan::EnvironmentMapStatistics& em = ls.environmentMap;
        o["environment"]                           = Json::object{{"valid", em.valid},
                                                                  {"last_capture_frame", double(em.lastCompleteFrame)},
                                                                  {"captures", double(em.captures)},
                                                                  {"progress", Json::array{em.progress, em.steps}},
                                                                  {"lut_computations", double(ls.lutComputations)}};
    }
    o["clouds"]     = Json::object{{"altitude", double(state.settings.clouds.altitude)},
                                   {"radius", double(state.settings.clouds.radius)},
                                   {"seed", double(state.settings.clouds.seed)},
                                   {"shadow_fade", double(state.settings.clouds.shadowFade)},
                                   {"fog_falloff", double(state.settings.clouds.fogFalloff)},
                                   {"enabled", state.parameters.clouds},
                                   {"drawn", ls.clouds},
                                   {"mean_coverage", double(ls.cloudMeanCoverage)}};
    o["settings"]   = keysJson(state, false);
    o["parameters"] = keysJson(state, true);
    // S36a1b: the debug state, the GPU sections (the last frame and the mean over the last 120 frames of the app's GPU
    // profile; the lighting sections against the §17.10 shadow budget, the debug sections separately), the lighting's
    // memory
    {
        Json::object d                    = lightingDebugJson(scene->lightingDebug()).object_items();
        d["frame"]                        = lightingDebugJson(scene->frameLightingDebug());
        d["freeze_cascades"]              = scene->cascadesFrozen();
        const std::optional<float> centre = scene->exposureViewCentreSetting();
        d["exposure_view"]                = Json::object{{"centre", centre ? Json(double(*centre)) : Json("auto")},
                                                         {"frame_centre", double(scene->frameExposureViewCentre())},
                                                         {"range", double(scene->exposureViewRangeSetting())}};
        d["views"]                        = Json(viewNames());
        d["overlay_names"]                = Json(overlayNames());
        d["mapping"]                      = mappingJson(app.lightingPreviewMapping());
        o["debug"]                        = d;
        Json::object g;
        Profiler& p     = app.gpuProfile();
        double lighting = 0, debug = 0;
        for (const std::string& n : lightingGpuSections())
        {
            const int id      = p.scope("gpu/" + n);
            double last       = 0;
            const bool has    = p.lastSample(id, last);
            const double mean = p.scopeStats(id, 120).avg;
            (lightingDebugGpuSection(n) ? debug : lighting) += mean;
            g[n] = Json::object{{"last_ms", has ? Json(last) : Json()}, {"mean_ms", mean}};
        }
        g["mean_frames"] = 120;
        g["lighting_ms"] = lighting;
        g["budget_ms"]   = 2.5;
        g["debug_ms"]    = debug;
        o["gpu"]         = g;
        o["memory"]      = Json::object{{"bytes", double(ls.memoryBytes)}};
    }
    // S36a2: the cascades of the newest recorded frame (instances and triangles of each cascade's last render, from
    // the GPU statistics)
    {
        Json::array list;
        for (int c = 0; c < ls.cascadeCount; ++c)
        {
            const Vulkan::CascadeStatistics& k = ls.cascades[size_t(c)];
            list.push_back(Json::object{{"cascade", c},
                                        {"valid", k.valid},
                                        {"split_near", double(k.splitNear)},
                                        {"split_far", double(k.splitFar)},
                                        {"center", vec3Json(k.center)},
                                        {"half_extent", double(k.halfExtent)},
                                        {"depth_range", double(k.depthRange)},
                                        {"texel_m", double(k.texelSize)},
                                        {"interval", k.interval},
                                        {"last_update", double(k.lastUpdate)},
                                        {"updated_this_frame", k.updatedThisFrame},
                                        {"instances", double(k.instances)},
                                        {"triangles", double(k.triangles)},
                                        {"terrain_nodes", double(k.terrainNodes)},
                                        {"terrain_triangles", double(k.terrainTriangles)},
                                        {"terrain_nodes_deduplicated", double(k.terrainDeduplicated)}});
        }
        const char* light = ls.shadowLight == 0   ? "none"
                            : !ls.physical        ? "studio"
                            : ls.shadowLight == 1 ? "sun"
                                                  : "moon";
        o["cascades"]     = Json::object{{"on", ls.shadows},
                                         {"enabled", state.parameters.sunShadows},
                                         {"frozen", ls.frozen},
                                         {"shadow_light", light},
                                         {"light_direction", vec3Json(ls.shadowLightDir)},
                                         {"count", ls.cascadeCount},
                                         {"resolution", state.parameters.cascadeResolution},
                                         {"linear_filter", ls.linearShadowFilter},
                                         {"memory_bytes", double(ls.shadowMemoryBytes)},
                                         {"list", list}};
    }
    // S36a2: the horizon map (the front map's transform, the map in progress, the blend)
    {
        const Vulkan::HorizonStatistics& h = ls.horizon;
        o["horizon"]                       = Json::object{{"on", h.on},
                                                          {"enabled", state.parameters.horizonShadows},
                                                          {"valid", h.valid},
                                                          {"resolution", h.resolution},
                                                          {"extent", h.extent},
                                                          {"texel_m", h.texel},
                                                          {"anchor", Json::array{h.corner(0), h.corner(1)}},
                                                          {"reference_height", h.reference},
                                                          {"direction", vec3Json(h.light)},
                                                          {"computing", h.computing},
                                                          {"progress", Json::array{h.rowsDone, h.resolution}},
                                                          {"rows_this_frame", h.rowsThisFrame},
                                                          {"blend", double(h.blend)},
                                                          {"recomputes", double(h.recomputes)},
                                                          {"completed", double(h.completed)},
                                                          {"last_complete_frame", double(h.lastComplete)},
                                                          {"level_at_10km", h.levelAt10km},
                                                          {"memory_bytes", double(h.memoryBytes)}};
    }
    // S36b: the local lights of the newest recorded frame
    {
        const Vulkan::LocalLightStatistics& l = ls.lights;
        o["lights"]                           = Json::object{{"on", l.on},
                                                             {"enabled", state.parameters.localLights},
                                                             {"clustered", state.parameters.clusteredLights},
                                                             {"submitted", double(l.submitted)},
                                                             {"candidates", double(l.candidates)},
                                                             {"visible", double(l.visible)},
                                                             {"over_budget", double(l.overBudget)},
                                                             {"shadowed", double(l.shadowed)},
                                                             {"without_tiles", double(l.withoutTiles)},
                                                             {"max_lights", state.parameters.maxLights},
                                                             {"memory_bytes", double(l.memoryBytes)}};
    }
    // S36b: the clusters of the newest completed frame
    {
        const Vulkan::LocalLightStatistics& l = ls.lights;
        const Vulkan::LightingParameters& p   = state.parameters;
        o["clusters"]                         = Json::object{{"grid", Json::array{p.clusterX, p.clusterY, p.clusterZ}},
                                                             {"cluster_near", double(p.clusterNear)},
                                                             {"cluster_far", double(p.clusterFar)},
                                                             {"max_lights_per_cluster", p.maxLightsPerCluster},
                                                             {"clustered", l.clustered},
                                                             {"clusters", double(l.clusterCount)},
                                                             {"max", double(l.clusterMax)},
                                                             {"mean", l.clusterMean},
                                                             {"overflowed", double(l.overflowedClusters)},
                                                             {"dropped_lights", double(l.droppedLights)},
                                                             {"frame", double(l.clusterFrame)},
                                                             {"memory_bytes", double(l.clusterMemory)}};
    }
    // S36b: the shadow atlas of the newest recorded frame
    {
        const Vulkan::AtlasStatistics& a = ls.atlas;
        o["atlas"]                       = Json::object{{"on", a.on},
                                                        {"resolution", a.resolution},
                                                        {"tile", a.tile},
                                                        {"tiles_total", double(a.tilesTotal)},
                                                        {"tiles_used", double(a.tilesUsed)},
                                                        {"static_renders", double(a.staticRenders)},
                                                        {"dynamic_renders", double(a.dynamicRenders)},
                                                        {"dynamic_clears", double(a.dynamicClears)},
                                                        {"hits", double(a.hits)},
                                                        {"queue", double(a.queue)},
                                                        {"dynamic_skipped", double(a.dynamicSkipped)},
                                                        {"terrain_nodes", double(a.terrainNodes)},
                                                        {"terrain_nodes_truncated", double(a.terrainNodesTruncated)},
                                                        {"lights_with_tiles", double(a.lightsWithTiles)},
                                                        {"lights_without_tiles", double(a.lightsWithoutTiles)},
                                                        {"faces_per_frame", a.facesPerFrame},
                                                        {"memory_bytes", double(a.memoryBytes)}};
    }
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
    return Json::object{
        {"time_of_day", hasWorldTime(w) ? Json(app.sampleViewEnvironment().time.timeOfDay) : Json()},
        // The sample's EV100 (as get_lighting's exposure.ev100: what the next frame uses)
        {"ev100", scene->hasEnvironment() ? Json(double(app.sampleViewEnvironment().state.ev100)) : Json()},
        {"tone_map", toneMapName(scene->parameters().toneMap)},
        {"debug_view", Vulkan::lightingDebugViewName(scene->lightingDebug().view)}};
}

void registerLightingTools(McpServer& server, EngineApp& app)
{
    server.addTool(
        "get_lighting",
        "Reads the lighting of the view camera's space: time (hours, day, rate; null without WorldTime), weather, sun "
        "and moon (direction, azimuth and elevation in degrees, illuminance at the top of the atmosphere and on the "
        "ground, the moon's phase), shadow_light, tone_map (identity = clay, which ignores time and weather), "
        "exposure (mode, source, ev100, curve_ev100, compensation, multiplier, effective: now, as the next frame "
        "uses them; frame_ev100, frame_effective: the last rendered frame's; pre_exposure; without an environment only "
        "the unit-less multiplier), ambient, clouds, settings (the set_lighting keys) and parameters "
        "(the LightingParameters). sections keeps only the named top-level keys. Answered at once. Changes nothing.",
        schema(
            {{"sections", arrayProperty("The top-level keys to return (default all)",
                                        McpServer::StringProperty("A top-level key of the answer", sectionNames()))}}),
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
        else if (n == "clouds" || n == "force_full_updates" || n == "sun_shadows" || n == "terrain_casts_in_cascades" ||
                 n == "horizon_shadows" || n == "local_lights" || n == "clustered_lights")
            p = McpServer::BooleanProperty(n);
        else if (n == "horizon_resolution")
            p = McpServer::IntegerProperty("A power of two in [128, 4096]");
        else if (n == "environment_resolution")
            p = McpServer::IntegerProperty("A power of two in [32, 512]");
        else if (n == "cloud_seed")
            p = McpServer::IntegerProperty(k.description);
        else if (n == "cascade_resolution")
            p = McpServer::IntegerProperty("A power of two in [256, 4096]");
        else if (n == "cascade_count" || n == "shadow_lod_bias" || n == "max_lights" || n == "cluster_x" ||
                 n == "cluster_y" || n == "cluster_z" || n == "max_lights_per_cluster" || n == "atlas_tile" ||
                 n == "faces_per_frame" || n == "horizon_rows_per_frame")
            p = McpServer::IntegerProperty(k.description + (n == "atlas_tile" ? " (divides atlas_resolution)" : ""));
        else if (n == "atlas_resolution")
            p = McpServer::IntegerProperty("A power of two in [512, 8192]");
        else if (n == "cascade_interval")
            p = arrayProperty("[4] frames between the cascades' updates (the first 1)",
                              Json::object{{"type", "integer"}});
        else
            p = McpServer::NumberProperty(k.description.empty() ? n : k.description);
        setProperties[n] = p;
    }
    server.addTool(
        "set_lighting",
        "Changes the lighting settings (not world state: not replicated, not saved), validated as a whole and applied "
        "all or nothing from the next frame: tone_map (aces, identity), exposure_mode (time_of_day, fixed), ev100, "
        "exposure_compensation (EV, +1 is twice as bright), ambient_scale, min_ambient ([r, g, b] lux), sun_override "
        "({azimuth_deg, elevation_deg} or null), cloud_altitude, cloud_radius, cloud_shadow_fade (m), cloud_seed (the "
        "coverage texture's), star_intensity, ground_albedo, rayleigh_scale, mie_scale, ozone_scale (the atmosphere's "
        "coefficients), fog_falloff (1/m), and the LightingParameters clouds, environment_resolution, "
        "environment_update_degrees, "
        "force_full_updates and (the cascades) sun_shadows, cascade_count (1..4), cascade_resolution (256..4096), "
        "shadow_distance (m), split_lambda, cascade_interval ([4], the first 1), depth_bias_texels, slope_bias, "
        "normal_offset_texels, cascade_blend, shadow_lod_bias, small_caster_texels, max_caster_height (m), "
        "min_shadow_elevation_deg, cascade_caster_distance (m), terrain_casts_in_cascades and (the far terrain horizon "
        "map) horizon_shadows, horizon_resolution (128..4096), horizon_extent (m), horizon_rows_per_frame and (S36b, "
        "the local lights) local_lights, clustered_lights (false: the brute-force reference), max_lights (1..4096; "
        "local_lights false switches them off), "
        "cluster_x, cluster_y (screen tiles, 1..64), cluster_z (depth slices, 2..128), cluster_near, cluster_far (m; "
        "cluster_near < cluster_far), max_lights_per_cluster (1..1024), atlas_resolution (512..8192, a power of two), "
        "atlas_tile (texels, divides atlas_resolution), faces_per_frame (0..11), atlas_depth_bias (D16 steps), "
        "atlas_normal_offset (texels). Returns "
        "values, previous "
        "(an undo payload of this tool) and memory_bytes (the cascades after the change; more than 1 GiB of lighting "
        "memory is refused). An error names every invalid key and changes nothing.",
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
            // The whole state (round 2, findings 2, 9: the shared check of the tool and the tab): the ranges, the
            // cross-key rules and the memory with the cluster buffers
            uint64_t memory = 0;
            if (errors.empty())
                validateLighting(app.scene()->lightingStatistics(), app.scene()->storageBufferLimit(), after.settings,
                                 after.parameters, errors, &memory);
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
            respond(ToolResult::Text(
                Json(Json::object{{"values", values}, {"previous", previous}, {"memory_bytes", double(memory)}})
                    .dump()));
        });

    // ---- S36a1b: the debug core
    server.addTool(
        "set_lighting_debug",
        "Changes the lighting's debug view (view: " + joined(viewNames()) +
            "; data views show raw values, a debug frame uses the identity tone map), the overlays (a list that "
            "replaces the set: " +
            joined(overlayNames()) +
            "), freeze_cascades (S36a2: the cascades and the camera they were fitted to stay, still rendered every "
            "frame; with the cascade_bounds and camera_frustum_splits overlays one flies around and sees what each "
            "covers), highlight_light (S36b: a light handle of list_lights, white in the light_volumes overlay and "
            "selected in the Lighting tab; 0 or null: none), the Lighting tab's preview mapping (mapping: {range: "
            "\"auto\" or [min, max], log, color_map: gray or turbo}) and the exposure view's heat map (exposure_view: "
            "{centre: EV100 or \"auto\", range: EV, 0.5..64; turbo over centre +- range / 2}). Missing arguments keep "
            "their value. Returns the state, the previous state and the known names. Changes only the rendering.",
        schema({{"view", McpServer::StringProperty("The debug view", viewNames())},
                {"overlays",
                 arrayProperty("The overlays (replaces the set)", McpServer::StringProperty("", overlayNames()))},
                {"freeze_cascades", McpServer::BooleanProperty("Freeze the cascades")},
                {"highlight_light",
                 McpServer::IntegerProperty("The highlighted light's handle (list_lights; 0 or null: none)")},
                {"mapping",
                 Json::object{{"type", "object"}, {"description", "{range: \"auto\" or [min, max], log, color_map}"}}},
                {"exposure_view",
                 Json::object{{"type", "object"}, {"description", "{centre: EV100 or \"auto\", range: EV}"}}}}),
        [&app](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(
                    a, {"view", "overlays", "mapping", "freeze_cascades", "highlight_light", "exposure_view"}, error))
                return respond(ToolResult::Error(error));
            const Json& hl = a["highlight_light"];
            if (!hl.is_null() && (!hl.is_number() || hl.number_value() != std::floor(hl.number_value()) ||
                                  hl.number_value() < 0 || hl.number_value() > 4294967295.0))
                return respond(ToolResult::Error("highlight_light: a light handle (an integer >= 0; 0 or null: none)"));
            if (!a["freeze_cascades"].is_null() && !a["freeze_cascades"].is_bool())
                return respond(ToolResult::Error("freeze_cascades: true or false"));
            Vulkan::RenderScene* scene = app.scene();
            if (!scene) return respond(ToolResult::Error(kNoDevice));
            const Vulkan::LightingDebugState before = scene->lightingDebug();
            Vulkan::LightingDebugState after        = before;
            Vulkan::LightingDebugState parsed;
            if (!parseLightingDebug(a["view"], a["overlays"], parsed, error))
            {
                // The argument is called view here (lighting_view in screenshot)
                if (error.rfind("lighting_view", 0) == 0) error = "view" + error.substr(13);
                return respond(ToolResult::Error(error));
            }
            if (!a["view"].is_null()) after.view = parsed.view;
            if (!a["overlays"].is_null()) after.overlays = parsed.overlays;
            if (a.object_items().count("highlight_light"))
                after.highlightLight = hl.is_null() ? 0u : uint32_t(hl.number_value());
            Vulkan::PreviewMapping mapping = app.lightingPreviewMapping();
            if (!a["mapping"].is_null())
            {
                const Json& m = a["mapping"];
                if (!m.is_object()) return respond(ToolResult::Error("mapping: {range, log, color_map}"));
                for (auto& kv : m.object_items())
                    if (kv.first != "range" && kv.first != "log" && kv.first != "color_map")
                        return respond(ToolResult::Error("mapping: unknown key \"" + kv.first +
                                                         "\" (keys: range, log, color_map)"));
                if (!parseMapping(m["range"], m["log"], m["color_map"], mapping, error))
                    return respond(ToolResult::Error("mapping." + error));
            }
            std::optional<float> centre = scene->exposureViewCentreSetting();
            float range                 = scene->exposureViewRangeSetting();
            if (!a["exposure_view"].is_null())
            {
                const Json& x = a["exposure_view"];
                if (!x.is_object()) return respond(ToolResult::Error("exposure_view: {centre, range}"));
                for (auto& kv : x.object_items())
                    if (kv.first != "centre" && kv.first != "range")
                        return respond(
                            ToolResult::Error("exposure_view: unknown key \"" + kv.first + "\" (keys: centre, range)"));
                if (x["centre"] == Json("auto"))
                    centre.reset();
                else if (!x["centre"].is_null())
                {
                    double c = 0;
                    if (!number(x["centre"], "exposure_view.centre", -30, 30, c, error))
                        return respond(ToolResult::Error(error + " or \"auto\""));
                    centre = float(c);
                }
                if (!x["range"].is_null())
                {
                    double r = 0;
                    if (!number(x["range"], "exposure_view.range", 0.5, 64, r, error))
                        return respond(ToolResult::Error(error));
                    range = float(r);
                }
            }
            const bool frozenBefore = scene->cascadesFrozen();
            if (!a["freeze_cascades"].is_null()) scene->setFreezeCascades(a["freeze_cascades"].bool_value());
            scene->setLightingDebug(after);
            scene->setExposureViewRange(centre, range);
            app.lightingPreviewMapping() = mapping;
            Json::object o               = lightingDebugJson(after).object_items();
            o["freeze_cascades"]         = scene->cascadesFrozen();
            Json::object prev            = lightingDebugJson(before).object_items();
            prev["freeze_cascades"]      = frozenBefore;
            o["previous"]                = prev;
            o["mapping"]                 = mappingJson(mapping);
            o["exposure_view"] =
                Json::object{{"centre", centre ? Json(double(*centre)) : Json("auto")}, {"range", range}};
            o["views"]         = Json(viewNames());
            o["overlay_names"] = Json(overlayNames());
            respond(ToolResult::Text(Json(o).dump()));
        });

    server.addTool(
        "capture_lighting",
        "Captures a lighting debug image as a picture plus a JSON text block: what it shows (the mapping range, log, "
        "colour map, the size, the mip) and the numbers of its source values (min, max, mean, count and a 16-bin "
        "histogram over the mapped range; with a manual range the edge bins also count the values below and above "
        "it). image: luminance (the resolved HDR luminance of the frame in cd/m^2, the display luminance in clay mode; "
        "log2 by default), brdf_lut (the split-sum BRDF LUT's scale A), transmittance_lut (the atmosphere's "
        "transmittance, luminance), multi_scattering_lut (the atmosphere's multiple-scattering transfer, luminance), "
        "sky_view_lut (the sky's radiance by azimuth and elevation, cd/m^2, log2 by default), environment (the "
        "environment map's front cube as a cross of the mip mip, cd/m^2, log2 by default), irradiance (its SH "
        "irradiance per direction, equirectangular, lux), cloud_coverage (the cloud layer's coverage at the "
        "cloudiness over its 16 km tile), cloud_shadow (the cloud shadow factor on the plane of the camera height "
        "around the camera, +- the shadow fade), cascade (S36a2: the cascades' light-space depth, 0 at the near end, "
        "1 at the far end, the minimum over the texels a preview texel covers; cascade: its index, without it every "
        "cascade side by side; the block names each cascade's world box and texel size), horizon (the front horizon "
        "map's shadow height, m relative to its reference; channel r: occluders from 64 m, g: beyond the cascades' "
        "terrain; the block names its anchor, extent and direction), clusters (S36b: the lights per cluster, 16 "
        "pixels per cluster; slice: a depth slice, without it the maximum over the slices; turbo from 0 by default), "
        "atlas (S36b: the shadow atlas's layer static or dynamic, the linear depth distance / range, at most 1024 "
        "texels; the block lists the tiles' owners) or atlas_tile (one light's tile at full resolution: light, face, "
        "layer). range (\"auto\" or [min, max]), log, color_map (gray, turbo) set the mapping; mip: the "
        "environment's mip (0..4, default 0; a request of its own, the Lighting tab's mip is not changed); stats_only: "
        "true returns the block without an image (the one path to the statistics of a GPU image); format png (default) "
        "or jpg, downscaled to max_width (default 1280, 64..8192); full: true updates incremental effects fully first "
        "(the environment map). Deferred: answered from the frame that computes the preview; an error when no frame "
        "is rendered before the deadline. Changes nothing.",
        schema(
            {{"image", McpServer::StringProperty("The image", captureImageNames())},
             {"layer",
              McpServer::StringProperty("atlas, atlas_tile: the layer (default static)", {"static", "dynamic"})},
             {"light", McpServer::IntegerProperty("atlas_tile: the light's handle (list_lights)")},
             {"face", McpServer::IntegerProperty("atlas_tile: the face (0..5 for a point light: +x, -x, +y, -y, +z, "
                                                 "-z; 0 for a spot light; default 0)")},
             {"slice", McpServer::IntegerProperty("clusters: the depth slice (0..cluster_z - 1; default the maximum "
                                                  "over the slices)")},
             {"channel", McpServer::StringProperty("horizon: the channel (default r)", {"r", "g"})},
             {"mip", McpServer::IntegerProperty("environment: the mip (0..4, default 0)")},
             {"cascade", McpServer::IntegerProperty("cascade: the cascade (0..count - 1; default all side by side)")},
             {"range", Json::object{{"description", "\"auto\" or [min, max]"}}},
             {"log", McpServer::BooleanProperty("log2 of the values before the mapping")},
             {"color_map", McpServer::StringProperty("The colour map", {"gray", "turbo"})},
             {"full", McpServer::BooleanProperty("Update incremental effects fully first")},
             {"max_width", McpServer::IntegerProperty("Downscale to at most this width (default 1280, 64..8192)")},
             {"format", McpServer::StringProperty("Image format (default png)", {"png", "jpg"})},
             {"stats_only", McpServer::BooleanProperty("Only the JSON block (no image)")}}),
        [&app, &server](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a,
                                {"image", "range", "log", "color_map", "full", "max_width", "format", "stats_only",
                                 "mip", "cascade", "channel", "slice", "layer", "light", "face"},
                                error))
                return respond(ToolResult::Error(error));
            Vulkan::RenderScene* scene = app.scene();
            if (!scene) return respond(ToolResult::Error(kNoDevice));
            std::optional<Vulkan::LightingImage> parsed =
                a["image"].is_string() ? Vulkan::lightingImageFromName(a["image"].string_value()) : std::nullopt;
            if (parsed == Vulkan::LightingImage::CascadeStrip) parsed.reset();  // "cascade" without the argument
            if (!parsed)
                return respond(ToolResult::Error(
                    "image: luminance, brdf_lut, transmittance_lut, multi_scattering_lut, sky_view_lut, environment, "
                    "irradiance, cloud_coverage, cloud_shadow, cascade, horizon, clusters, atlas or atlas_tile"));
            Vulkan::LightingImage image = *parsed;
            // S36a2: the cascades (one, or all side by side)
            int cascade = -1;
            if (image == Vulkan::LightingImage::Cascade)
            {
                const Vulkan::LightingStatistics& ls = scene->lightingStatistics();
                if (!ls.shadows || ls.cascadeCount == 0)
                    return respond(ToolResult::Error(
                        "cascade: no cascades are rendered (set_lighting sun_shadows: true, a shadow light)"));
                if (!a["cascade"].is_null())
                {
                    const double c = a["cascade"].number_value();
                    if (!a["cascade"].is_number() || c != std::floor(c) || c < 0 || c >= ls.cascadeCount)
                        return respond(
                            ToolResult::Error("cascade: an integer 0.." + std::to_string(ls.cascadeCount - 1)));
                    cascade = int(c);
                }
                else
                    image = Vulkan::LightingImage::CascadeStrip;
            }
            else if (!a["cascade"].is_null())
                return respond(ToolResult::Error("cascade: only for image cascade"));
            // S36b: the shadow atlas (a layer) and one light's tile
            int atlasLayer = 0, face = 0;
            uint32_t lightHandle  = 0;
            const bool atlasImage = image == Vulkan::LightingImage::Atlas || image == Vulkan::LightingImage::AtlasTile;
            if (atlasImage)
            {
                if (!scene->lightingStatistics().atlas.on)
                    return respond(ToolResult::Error(
                        std::string(Vulkan::lightingImageName(image)) +
                        ": no shadow atlas (set_lighting local_lights: true, a light with castShadows in view)"));
                if (!a["layer"].is_null())
                {
                    if (a["layer"] != Json("static") && a["layer"] != Json("dynamic"))
                        return respond(ToolResult::Error("layer: static or dynamic"));
                    atlasLayer = a["layer"] == Json("dynamic") ? 1 : 0;
                }
                if (image == Vulkan::LightingImage::Atlas)
                {
                    if (!a["light"].is_null() || !a["face"].is_null())
                        return respond(ToolResult::Error("light, face: only for image atlas_tile"));
                    scene->setAtlasPreview(atlasLayer);
                }
                else
                {
                    const double h = a["light"].number_value();
                    if (!a["light"].is_number() || h != std::floor(h) || h < 1)
                        return respond(
                            ToolResult::Error("light: the handle of a light with atlas tiles (list_lights)"));
                    lightHandle                  = uint32_t(h);
                    const std::vector<int> tiles = scene->shadowAtlas().tilesOf(lightHandle);
                    if (tiles.empty())
                        return respond(ToolResult::Error("light: light " + std::to_string(lightHandle) +
                                                         " has no atlas tiles (not shadowed, not visible or over the "
                                                         "tile budget)"));
                    if (!a["face"].is_null())
                    {
                        const double f = a["face"].number_value();
                        if (!a["face"].is_number() || f != std::floor(f) || f < 0 || f >= double(tiles.size()))
                            return respond(
                                ToolResult::Error("face: an integer 0.." + std::to_string(tiles.size() - 1)));
                        face = int(f);
                    }
                    scene->setAtlasTilePreview(lightHandle, face, atlasLayer);
                }
            }
            else if (!a["layer"].is_null() || !a["light"].is_null() || !a["face"].is_null())
                return respond(ToolResult::Error("layer, light, face: only for images atlas and atlas_tile"));
            // S36b: the clusters (a slice or the maximum over the slices)
            int slice = -1;
            if (image == Vulkan::LightingImage::Clusters)
            {
                if (scene->lightingStatistics().lights.visible == 0)
                    return respond(ToolResult::Error(
                        "clusters: the frame has no local lights (set_lighting local_lights: true, lights in view)"));
                if (!a["slice"].is_null())
                {
                    const double z = a["slice"].number_value();
                    const int n    = scene->frameClusterGrid().z;
                    if (!a["slice"].is_number() || z != std::floor(z) || z < 0 || z >= n)
                        return respond(ToolResult::Error("slice: an integer 0.." + std::to_string(n - 1)));
                    slice = int(z);
                }
                scene->setClusterPreviewSlice(slice);
            }
            else if (!a["slice"].is_null())
                return respond(ToolResult::Error("slice: only for image clusters"));
            // S36a2: the horizon map's channel
            if (image == Vulkan::LightingImage::Horizon)
            {
                const Vulkan::HorizonStatistics& h = scene->lightingStatistics().horizon;
                if (!h.valid)
                    return respond(ToolResult::Error(
                        "horizon: no horizon map (set_lighting horizon_shadows: true, a shadow light, terrain)"));
                if (!a["channel"].is_null() && a["channel"] != Json("r") && a["channel"] != Json("g"))
                    return respond(ToolResult::Error("channel: r or g"));
                if (a["full"].bool_value()) scene->requestHorizonRecompute(true);
            }
            else if (!a["channel"].is_null())
                return respond(ToolResult::Error("channel: only for image horizon"));
            int mip = 0;
            if (!a["mip"].is_null())
            {
                const double m = a["mip"].number_value();
                if (!a["mip"].is_number() || m != std::floor(m) || m < 0 || m > Vulkan::EnvironmentMap::kMips - 1)
                    return respond(ToolResult::Error("mip: an integer 0..4"));
                if (image != Vulkan::LightingImage::Environment)
                    return respond(ToolResult::Error("mip: only for image environment"));
                mip = int(m);
            }
            Vulkan::PreviewMapping mapping;
            mapping.log = image == Vulkan::LightingImage::Luminance || image == Vulkan::LightingImage::SkyViewLut ||
                          image == Vulkan::LightingImage::Environment;
            if (!parseMapping(a["range"], a["log"], a["color_map"], mapping, error))
                return respond(ToolResult::Error(error));
            mapping.mip = mip;
            // The cascade's layer or the horizon map's channel of this request (not the tab's)
            if (image == Vulkan::LightingImage::Cascade) mapping.layer = cascade;
            if (image == Vulkan::LightingImage::Horizon) mapping.layer = a["channel"] == Json("g") ? 1 : 0;
            for (const char* b : {"full", "stats_only"})
                if (!a[b].is_null() && !a[b].is_bool())
                    return respond(ToolResult::Error(std::string(b) + ": true or false"));
            const bool statsOnly = a["stats_only"].bool_value();
            if (a["full"].bool_value()) scene->requestFullUpdate();
            const std::string format = a["format"].is_null() ? "png" : a["format"].string_value();
            if (format != "png" && format != "jpg") return respond(ToolResult::Error("format: png or jpg"));
            int maxWidth = 1280;
            if (!a["max_width"].is_null())
            {
                const double w = a["max_width"].number_value();
                if (!a["max_width"].is_number() || w != std::floor(w) || w < 64 || w > 8192)
                    return respond(ToolResult::Error("max_width: an integer 64..8192"));
                maxWidth = int(w);
            }
            if (image == Vulkan::LightingImage::Clusters && a["range"].is_null())
            {
                // Lights per cluster: from 0 (turbo by default)
                mapping.autoRange = false;
                mapping.min       = 0;
                mapping.max       = std::max(float(scene->lightingStatistics().lights.clusterMax), 1.0f);
                if (a["color_map"].is_null()) mapping.colorMap = Vulkan::ColorMap::Turbo;
            }
            auto answered          = std::make_shared<bool>(false);
            const auto deadline    = answerDeadline(server);
            const std::string name = Vulkan::lightingImageName(image);
            scene->lightingDebugImages().request(
                image, mapping,
                [&app, respond, answered, statsOnly, format, maxWidth, mapping, name, cascade, slice, atlasLayer,
                 lightHandle, face](const Vulkan::LightingDebug::Capture& c)
                {
                    if (*answered) return;
                    *answered = true;
                    // A cascade count lowered after the request: the frame converted a substitute layer
                    const int count = app.scene() ? app.scene()->lightingStatistics().cascadeCount : 0;
                    if (cascade >= count)
                        return respond(
                            ToolResult::Error("cascade: the cascade count changed to " + std::to_string(count)));
                    const Vulkan::PreviewStatistics& s = c.statistics;
                    Json::array histogram;
                    for (uint32_t h : s.histogram) histogram.push_back(double(h));
                    Vulkan::PreviewMapping used = mapping;
                    used.autoRange              = false;
                    used.min                    = s.rangeMin;
                    used.max                    = s.rangeMax;
                    Json::object block{{"image", name},
                                       {"size", Json::array{c.size(0), c.size(1)}},
                                       {"frame", double(app.scene() ? app.scene()->frames() + 1 : 0)},
                                       {"mapping", mappingJson(used)},
                                       {"auto_range", mapping.autoRange},
                                       {"stats", Json::object{{"min", double(s.min)},
                                                              {"max", double(s.max)},
                                                              {"mean", double(s.mean)},
                                                              {"count", double(s.count)},
                                                              {"histogram", histogram}}}};
                    if (name == "luminance")
                        block["unit"] = mapping.log ? "log2 cd/m^2 (clay: log2 of the display luminance)"
                                                    : "cd/m^2 (clay: the display luminance)";
                    if (name == "sky_view_lut" || name == "environment")
                        block["unit"] = mapping.log ? "log2 cd/m^2" : "cd/m^2";
                    if (name == "environment") block["mip"] = mapping.mip;
                    if (name == "multi_scattering_lut")
                        block["unit"] = "multiple-scattering transfer (luminance of rgb, per unit illuminance)";
                    if (name == "irradiance") block["unit"] = mapping.log ? "log2 lux" : "lux";
                    if (name == "transmittance_lut") block["unit"] = "transmittance (luminance of rgb)";
                    if (name == "cloud_coverage") block["unit"] = "coverage (opacity 0..1)";
                    if (name == "cloud_shadow") block["unit"] = "the shadow light's factor (0..1)";
                    if (name == "atlas" || name == "atlas_tile")
                    {
                        // S36b: the layer, the tiles' owners (light handle, face), their last renders
                        block["unit"]  = "linear depth (distance / range; 1: nothing)";
                        block["layer"] = atlasLayer == 0 ? "static" : "dynamic";
                        if (Vulkan::RenderScene* sc = app.scene())
                        {
                            Vulkan::ShadowAtlas& at = sc->shadowAtlas();
                            Json::array tiles;
                            for (int t = 0; t < int(at.tiles().size()); ++t)
                            {
                                const Vulkan::AtlasTileInfo& ti = at.tiles()[size_t(t)];
                                if (!ti.owner) continue;
                                if (name == "atlas_tile" && ti.owner != lightHandle) continue;
                                tiles.push_back(
                                    Json::object{{"tile", t},
                                                 {"light", double(ti.owner)},
                                                 {"face", ti.face},
                                                 {"origin", Json::array{(t % at.tilesPerSide()) * at.tileSize(),
                                                                        (t / at.tilesPerSide()) * at.tileSize()}},
                                                 {"static_valid", ti.staticValid},
                                                 {"static_frame", double(ti.staticFrame)},
                                                 {"dynamic_content", ti.dynamicContent}});
                            }
                            block["tiles"]      = tiles;
                            block["resolution"] = at.resolution();
                            block["tile_size"]  = at.tileSize();
                            if (name == "atlas_tile")
                            {
                                block["light"] = double(lightHandle);
                                block["face"]  = face;
                            }
                        }
                    }
                    if (name == "clusters")
                    {
                        // S36b: the grid, the slice and its depth range
                        block["unit"] = "lights per cluster (after the cap)";
                        if (Vulkan::RenderScene* sc = app.scene())
                        {
                            const Vulkan::ClusterGrid& g = sc->frameClusterGrid();
                            Json::object cl{{"grid", Json::array{g.x, g.y, g.z}},
                                            {"cell_pixels", Vulkan::LightClusters::kCellPixels},
                                            {"slice", slice >= 0 ? Json(slice) : Json("max")},
                                            {"max_lights_per_cluster", g.maxLightsPerCluster}};
                            if (slice >= 0)
                            {
                                double d0, d1;
                                g.sliceDepth(slice, d0, d1);
                                cl["depth"] = Json::array{d0, d1};
                            }
                            const Vulkan::LocalLightStatistics& l = sc->lightingStatistics().lights;
                            cl["overflowed"]                      = double(l.overflowedClusters);
                            block["clusters"]                     = cl;
                        }
                    }
                    if (name == "cascade" || name == "cascade_strip")
                    {
                        // The cascades' boxes (world) and texel sizes of the frame
                        block["image"] = "cascade";
                        block["unit"]  = "light-space depth (0 near end, 1 far end; 1: nothing)";
                        Json::array boxes;
                        if (Vulkan::RenderScene* sc = app.scene())
                        {
                            const Vulkan::LightingStatistics& ls = sc->lightingStatistics();
                            for (int k = 0; k < ls.cascadeCount; ++k)
                            {
                                if (cascade >= 0 && k != cascade) continue;
                                const Vulkan::CascadeStatistics& cs = ls.cascades[size_t(k)];
                                boxes.push_back(
                                    Json::object{{"cascade", k},
                                                 {"center", vec3Json(cs.center)},
                                                 {"half_extent", double(cs.halfExtent)},
                                                 {"depth_range", double(cs.depthRange)},
                                                 {"texel_m", double(cs.texelSize)},
                                                 {"split", Json::array{double(cs.splitNear), double(cs.splitFar)}}});
                            }
                            block["light_direction"] = vec3Json(ls.shadowLightDir);
                        }
                        block["cascades"] = boxes;
                        block["layout"]   = cascade >= 0 ? "one cascade" : "every cascade side by side (0 left)";
                    }
                    if (name == "horizon")
                    {
                        block["unit"] = "shadow height (m) relative to the map's reference height";
                        if (Vulkan::RenderScene* sc = app.scene())
                        {
                            // (the map the frame converted, after its update)
                            const Vulkan::HorizonStatistics& h = sc->lightingStatistics().horizonPreview;
                            block["channel"]                   = mapping.layer == 1 ? "g" : "r";
                            block["anchor"]                    = Json::array{h.corner(0), h.corner(1)};
                            block["extent"]                    = h.extent;
                            block["texel_m"]                   = h.texel;
                            block["reference_height"]          = h.reference;
                            block["direction"]                 = vec3Json(h.light);
                        }
                    }
                    if (statsOnly || c.image.width == 0) return respond(ToolResult::Text(Json(block).dump()));
                    TemplatedImage<ucvec4> img = c.image;
                    const int w                = std::min(img.w, maxWidth);
                    const int h                = std::max(1, int(std::lround(double(img.h) * w / std::max(1, img.w))));
                    TemplatedImage<ucvec3> out(h, w);
                    for (int y = 0; y < h; ++y)
                        for (int x = 0; x < w; ++x)
                        {
                            const ucvec4 p =
                                img(std::min(img.h - 1, y * img.h / h), std::min(img.w - 1, x * img.w / w));
                            out(y, x) = ucvec3(p(0), p(1), p(2));
                        }
                    out.set_compression_quality(90);
                    const auto data = out.saveToMemory(format);
                    if (data.empty())
                        return respond(ToolResult::Error("capture_lighting: could not encode as " + format));
                    ToolResult result = ToolResult::Image(data, format == "png" ? "image/png" : "image/jpeg");
                    result.addText(Json(block).dump());
                    respond(result);
                },
                !statsOnly);
            server.addPollListener(
                [answered, respond, deadline]()
                {
                    if (*answered) return false;
                    if (std::chrono::steady_clock::now() < deadline) return true;
                    *answered = true;
                    respond(
                        ToolResult::Error("capture_lighting: no frame was rendered before the deadline (rendering "
                                          "disabled or the window minimized)"));
                    return false;
                });
        });

    server.addTool(
        "probe_lighting",
        "Reads the lighting at one pixel ([x, y], row 0 at the top) or at the screen position of a world point "
        "(position [x, y, z]): a sequence of 11 frames (frame 1 normal, then the albedo, normal and material views, "
        "the direct, ambient and specular radiance, the cloud shadow, the cascades, the horizon map and (S36b) the "
        "local lights' atlas shadow), each copying "
        "the resolved pixel (the MSAA "
        "average); the camera "
        "and "
        "the environment are held during it. Answers depth, world position and distance, albedo, roughness, "
        "metalness, ao, normal, the shadow factors (cloud, cascade with the cascade index, null outside every cascade, "
        "horizon with the horizon map's shadow heights R and G above the receiver in m, and their product), the "
        "probe pixel should lie inside a surface: the values are the MSAA average (at a cascade border or an edge the "
        "cascade index is rounded from the average); R and G pass the fp16 target (about 1 m precision at 1-2 km, "
        "2 m up to 4 km, 32 m near the +-60 km clamp), the "
        "direct, ambient (diffuse) and specular "
        "(environment) radiance and the frame's radiance (cd/m^2; clay: display values), its luminance and EV100, "
        "frames and consistent (false when the depth changed by more than 1% between the frames; the camera and the "
        "environment are held); S36b: atlas_shadow (the darkest atlas factor of the shadowed local lights that reach "
        "the pixel and their count), cluster (its index), cluster_lights (handle, distance and reflected radiance of "
        "each light of the cluster; without the shadow), cluster_total. A position is projected with the camera of the "
        "sequence's first frame. Screenshots and lighting captures wait for the frame after the sequence; the window "
        "shows the first frame meanwhile. Deferred; an error when no frames are rendered before the deadline. Changes "
        "nothing.",
        schema({{"pixel", arrayProperty("[x, y] in pixels", Json::object{{"type", "integer"}})},
                {"position", McpServer::Vec3Property("A world point [x, y, z] (its screen position)")}}),
        [&app, &server](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a, {"pixel", "position"}, error)) return respond(ToolResult::Error(error));
            Vulkan::RenderScene* scene = app.scene();
            if (!scene) return respond(ToolResult::Error(kNoDevice));
            const ivec2 size = scene->size();
            if (a["pixel"].is_null() == a["position"].is_null())
                return respond(ToolResult::Error("probe_lighting: pixel [x, y] or position [x, y, z] (one of them)"));
            ivec2 pixel(0, 0);
            std::optional<Vec3> world;
            if (!a["pixel"].is_null())
            {
                const Json& p = a["pixel"];
                if (!p.is_array() || p.array_items().size() != 2 || !p[0].is_number() || !p[1].is_number() ||
                    p[0].number_value() != std::floor(p[0].number_value()) ||
                    p[1].number_value() != std::floor(p[1].number_value()))
                    return respond(ToolResult::Error("pixel: [x, y] (integers)"));
                pixel = ivec2(int(p[0].number_value()), int(p[1].number_value()));
            }
            else
            {
                const Json& p = a["position"];
                if (!p.is_array() || p.array_items().size() != 3)
                    return respond(ToolResult::Error("position: [x, y, z]"));
                Vec3 w;
                for (int i = 0; i < 3; ++i)
                {
                    double v = 0;
                    if (!number(p[size_t(i)], "position[" + std::to_string(i) + "]", -1e9, 1e9, v, error))
                        return respond(ToolResult::Error(error));
                    w(i) = v;
                }
                // Projected with the camera of the sequence's frame 1 (RenderScene::requestProbe)
                world = w;
            }
            if (!world && (pixel(0) < 0 || pixel(1) < 0 || pixel(0) >= size(0) || pixel(1) >= size(1)))
                return respond(ToolResult::Error("pixel: (" + std::to_string(pixel(0)) + ", " +
                                                 std::to_string(pixel(1)) + ") is outside the image (" +
                                                 std::to_string(size(0)) + " x " + std::to_string(size(1)) + ")"));
            auto answered       = std::make_shared<bool>(false);
            const auto deadline = answerDeadline(server);
            auto answer         = [respond, answered, world](const Vulkan::LightingProbe& p)
            {
                if (*answered) return;
                *answered = true;
                if (world && !p.inImage)
                    return respond(ToolResult::Error("position: outside the image or behind the camera (pixel (" +
                                                     std::to_string(p.pixel(0)) + ", " + std::to_string(p.pixel(1)) +
                                                     "))"));
                Json::object o{{"pixel", Json::array{p.pixel(0), p.pixel(1)}},
                               {"frames", p.frames},
                               {"consistent", p.consistent},
                               {"physical", p.physical},
                               {"hit", p.hit},
                               {"depth", double(p.depth)},
                               {"color", vec3Json(p.color)},
                               {"luminance", double(p.luminance)},
                               {"ev100", double(p.ev100)},
                               {"note",
                                "the resolved pixel: the MSAA average (frame 1 in ACES with MSAA: the tone-map "
                                "weighted resolve, which weights bright samples less at edges)"}};
                if (p.hit)
                {
                    o["position"] = vec3Json(p.position);
                    o["distance"] = p.distance;
                }
                o["albedo"]    = vec3Json(p.albedo);
                o["normal"]    = vec3Json(p.normal);
                o["roughness"] = double(p.roughness);
                o["metalness"] = double(p.metalness);
                o["ao"]        = double(p.ao);
                o["shadow"]    = Json::object{{"cloud", double(p.cloudShadow)},
                                              {"cascade", double(p.cascadeShadow)},
                                              {"cascade_index", p.cascade >= 0 ? Json(p.cascade) : Json()},
                                              {"horizon", double(p.horizonShadow)},
                                              {"horizon_r_above_receiver", double(p.horizonR)},
                                              {"horizon_g_above_receiver", double(p.horizonG)},
                                              {"factor", double(p.shadowFactor)}};
                o["radiance"]  = Json::object{{"direct", vec3Json(p.direct)},
                                              {"ambient", vec3Json(p.ambient)},
                                              {"specular", vec3Json(p.specular)}};
                // S36b: the cluster and its lights with their contributions
                Json::array lights;
                for (const auto& l : p.lights)
                    lights.push_back(Json::object{
                        {"handle", double(l.handle)}, {"distance", l.distance}, {"radiance", vec3Json(l.radiance)}});
                o["atlas_shadow"]   = Json::object{{"factor", double(p.atlasShadow)}, {"lights", p.atlasShadowLights}};
                o["cluster"]        = p.cluster >= 0 ? Json(p.cluster) : Json();
                o["cluster_lights"] = lights;
                o["cluster_total"]  = double(p.clusterTotal);
                respond(ToolResult::Text(Json(o).dump()));
            };
            if (world)
                scene->requestProbe(*world, answer);
            else
                scene->requestProbe(pixel, answer);
            server.addPollListener(
                [answered, respond, deadline]()
                {
                    if (*answered) return false;
                    if (std::chrono::steady_clock::now() < deadline) return true;
                    *answered = true;
                    respond(
                        ToolResult::Error("probe_lighting: the frames were not rendered before the deadline "
                                          "(rendering disabled or the window minimized)"));
                    return false;
                });
        });
    server.addTool(
        "list_lights",
        "Reads the local lights (S36b): the scene's lights (entity lights of LightSource components and lights added "
        "on "
        "the scene, e.g. S30's piece lights) ordered by handle, then the entity lights that are not rendered (switched "
        "off, invalid or without a pose) ordered by entity index. Each: handle (null when not rendered), entity (null "
        "for a scene light), rendered and state (rendered, off, \"invalid: <reason>\", no pose), kind (point or spot), "
        "position (world), direction (spots), color (linear), lumen, range, inner and outer cone angles (degrees), "
        "shadowed (castShadows), visible this frame (selected within the budget), distance to the camera, the "
        "importance, the clusters it touches, its atlas tiles and their last render frame, the cache state. Filters: "
        "near {position: [x, y, z], radius} (the light's sphere intersects), shadowed, visible, rendered. Paged: "
        "limit (default 100, at most 1000), after (the last handle of the previous page: the answer's next) or "
        "after_entity (the last entity index of the previous page among the unrendered lights: the answer's "
        "next_entity); matched_total counts every light that matches the filters. An off light is switched on with "
        "set_component LightSource {on: true} (a game's systems may switch it again, e.g. game_world's lamps follow "
        "the time of day); scene lights are read-only. Changes nothing.",
        schema(
            {{"near", Json::object{{"type", "object"},
                                   {"description", "{position: [x, y, z], radius}: lights whose sphere intersects"}}},
             {"shadowed", McpServer::BooleanProperty("Only lights with (true) or without (false) castShadows")},
             {"visible", McpServer::BooleanProperty("Only lights in (true) or not in (false) this frame")},
             {"rendered",
              McpServer::BooleanProperty("Only lights the scene has (true) or entity lights it does not have (false: "
                                         "off, invalid, no pose)")},
             {"limit", McpServer::IntegerProperty("At most this many lights (default 100, 1..1000)")},
             {"after", McpServer::IntegerProperty("Rendered lights with a handle after this one (paging)")},
             {"after_entity", McpServer::IntegerProperty("Only unrendered lights with an entity index after this one "
                                                         "(paging, the answer's next_entity)")}}),
        [&app](const Json& a, McpServer::Respond respond)
        {
            std::string error;
            if (unknownArgument(a, {"near", "shadowed", "visible", "rendered", "limit", "after", "after_entity"},
                                error))
                return respond(ToolResult::Error(error));
            Vulkan::RenderScene* scene = app.scene();
            if (!scene) return respond(ToolResult::Error(kNoDevice));
            double limit = 100, after = 0, afterEntity = -1;
            if (!a["limit"].is_null() && !number(a["limit"], "limit", 1, 1000, limit, error))
                return respond(ToolResult::Error(error));
            if (!a["after"].is_null() && !number(a["after"], "after", 0, 4294967295.0, after, error))
                return respond(ToolResult::Error(error));
            if (!a["after_entity"].is_null() &&
                !number(a["after_entity"], "after_entity", 0, 4294967295.0, afterEntity, error))
                return respond(ToolResult::Error(error));
            if (limit != std::floor(limit) || after != std::floor(after) || afterEntity != std::floor(afterEntity))
                return respond(ToolResult::Error("limit, after and after_entity: integers"));
            if (!a["after"].is_null() && !a["after_entity"].is_null())
                return respond(ToolResult::Error("after or after_entity, not both"));
            for (const char* b : {"shadowed", "visible", "rendered"})
                if (!a[b].is_null() && !a[b].is_bool())
                    return respond(ToolResult::Error(std::string(b) + ": true or false"));
            std::optional<std::pair<Vec3, double>> near;
            if (!a["near"].is_null())
            {
                const Json& n = a["near"];
                const Json& p = n["position"];
                if (!n.is_object() || !p.is_array() || p.array_items().size() != 3 || !n["radius"].is_number())
                    return respond(ToolResult::Error("near: {position: [x, y, z], radius}"));
                for (auto& kv : n.object_items())
                    if (kv.first != "position" && kv.first != "radius")
                        return respond(ToolResult::Error("near: unknown key \"" + kv.first + "\" (position, radius)"));
                Vec3 pos;
                for (int i = 0; i < 3; ++i)
                {
                    if (!p[size_t(i)].is_number() || !std::isfinite(p[size_t(i)].number_value()))
                        return respond(ToolResult::Error("near.position: three finite numbers"));
                    pos(i) = p[size_t(i)].number_value();
                }
                double r = 0;
                if (!number(n["radius"], "near.radius", 0, 1e7, r, error)) return respond(ToolResult::Error(error));
                near = std::make_pair(pos, r);
            }
            // The entity lights (the view world's extract: the client world with --coop), rendered or not
            std::unordered_map<uint32_t, Entity> entities;
            std::vector<ExtractSystem::LightEntityState> unrendered;
            for (ExtractSystem::LightEntityState& st : app.world().extract().lightStates())
            {
                if (st.handle)
                    entities[st.handle.id] = st.entity;
                else
                    unrendered.push_back(std::move(st));
            }
            std::sort(unrendered.begin(), unrendered.end(),
                      [](const auto& x, const auto& y) { return x.entity.index < y.entity.index; });
            std::unordered_map<uint32_t, const Vulkan::FrameLightInfo*> frame;
            for (const Vulkan::FrameLightInfo& f : scene->frameLights()) frame[f.handle] = &f;
            std::vector<const Vulkan::LocalLights::Light*> list;
            for (const Vulkan::LocalLights::Light& l : scene->localLights().lights()) list.push_back(&l);
            std::sort(list.begin(), list.end(), [](auto* x, auto* y) { return x->handle.id < y->handle.id; });
            const Vec3 camera = app.viewCamera().position;
            auto deg          = [](float c) { return std::acos(std::clamp(double(c), -1.0, 1.0)) * 57.29577951308232; };
            auto passes       = [&](const Vec3* p, double range, bool shadowed, bool visible, bool rendered)
            {
                if (near && (!p || (*p - near->first).norm() > near->second + range)) return false;
                if (!a["shadowed"].is_null() && a["shadowed"].bool_value() != shadowed) return false;
                if (!a["visible"].is_null() && a["visible"].bool_value() != visible) return false;
                if (!a["rendered"].is_null() && a["rendered"].bool_value() != rendered) return false;
                return true;
            };
            Json::array out;
            Json next, nextEntity;
            size_t matched = 0;
            // The scene's lights (by handle; none after an after_entity cursor)
            for (const Vulkan::LocalLights::Light* l : list)
            {
                if (afterEntity >= 0) break;
                const Vulkan::LightData& d = l->data;
                const Vec3 p               = l->worldPosition();
                auto f                     = frame.find(l->handle.id);
                const bool visible         = f != frame.end();
                if (!passes(&p, double(d.range), d.castShadows, visible, true)) continue;
                matched++;
                if (l->handle.id <= uint32_t(after)) continue;
                if (out.size() >= size_t(limit))
                {
                    if (next.is_null() && nextEntity.is_null()) next = out.back()["handle"];
                    continue;
                }
                auto e = entities.find(l->handle.id);
                Json::object o{{"handle", double(l->handle.id)},
                               {"entity", e != entities.end() ? WorldApi::entityJson(e->second) : Json()},
                               {"list", l->list ? Json(double(l->list)) : Json()},
                               {"rendered", true},
                               {"state", "rendered"},
                               {"kind", Vulkan::lightKindName(d.kind)},
                               {"space", int(l->chunk.space)},
                               {"position", vec3Json(p)},
                               {"color", vec3Json(d.color)},
                               {"lumen", double(d.lumen)},
                               {"range", double(d.range)},
                               {"priority", double(d.priority)},
                               {"shadowed", d.castShadows},
                               {"visible", visible},
                               {"distance", (p - camera).norm()}};
                if (d.kind == Vulkan::LightKind::Spot)
                {
                    o["direction"]       = vec3Json(d.direction);
                    o["inner_angle_deg"] = deg(d.cosInner);
                    o["outer_angle_deg"] = deg(d.cosOuter);
                }
                if (d.flicker.amplitude > 0)
                    o["flicker"] = Json::object{{"amplitude", double(d.flicker.amplitude)},
                                                {"frequency", double(d.flicker.frequency)}};
                if (visible)
                {
                    const Vulkan::FrameLightInfo& fi = *f->second;
                    o["index"]                       = double(fi.index);
                    o["importance"]                  = fi.importance;
                    o["clusters"]                    = double(fi.clusters);
                    o["atlas"]                       = Json::object{{"tiles", fi.tiles},
                                                                    {"first_tile", fi.firstTile >= 0 ? Json(fi.firstTile) : Json()},
                                                                    {"last_render_frame", double(fi.lastRender)},
                                                                    {"ready", fi.shadowReady}};
                }
                out.push_back(o);
            }
            // The entity lights the scene does not have (by entity index)
            for (const ExtractSystem::LightEntityState& st : unrendered)
            {
                const LightSource& src = st.source;
                Vec3 p                 = Vec3::Zero();
                if (st.hasPose) p = st.pose.position + st.pose.rotation * (src.offset * st.pose.scale);
                if (!passes(st.hasPose ? &p : nullptr, src.range, src.castShadows, false, false)) continue;
                matched++;
                if (afterEntity >= 0 && double(st.entity.index) <= afterEntity) continue;
                if (out.size() >= size_t(limit))
                {
                    if (next.is_null() && nextEntity.is_null())
                    {
                        const Json& last = out.back();
                        if (last["handle"].is_null())
                            nextEntity = last["entity"]["index"];
                        else
                            next = last["handle"];
                    }
                    continue;
                }
                Json::object o{{"handle", Json()},
                               {"entity", WorldApi::entityJson(st.entity)},
                               {"list", Json()},
                               {"rendered", false},
                               {"state", st.state},
                               {"on", src.on},
                               {"kind", Vulkan::lightKindName(src.kind)},
                               {"space", st.hasPose ? Json(int(st.pose.space)) : Json()},
                               {"position", st.hasPose ? vec3Json(p) : Json()},
                               {"color", vec3Json(src.color)},
                               {"lumen", src.lumen},
                               {"range", src.range},
                               {"priority", src.priority},
                               {"shadowed", src.castShadows},
                               {"visible", false},
                               {"distance", st.hasPose ? Json((p - camera).norm()) : Json()}};
                if (src.kind == Vulkan::LightKind::Spot)
                {
                    o["direction"] =
                        st.hasPose ? vec3Json(Vec3((st.pose.rotation * src.rotation * Vec3(0, 0, -1)))) : Json();
                    o["inner_angle_deg"] = src.innerAngleDeg;
                    o["outer_angle_deg"] = src.outerAngleDeg;
                }
                out.push_back(o);
            }
            Json::object answer{{"lights", out},
                                {"count", double(out.size())},
                                {"total", double(scene->localLights().size() + unrendered.size())},
                                {"matched_total", double(matched)},
                                {"unrendered", double(unrendered.size())},
                                {"frame", double(scene->lightingStatistics().frame)},
                                {"next", next},
                                {"next_entity", nextEntity}};
            respond(ToolResult::Text(Json(answer).dump()));
        });
}

}  // namespace Engine
}  // namespace Saiga
