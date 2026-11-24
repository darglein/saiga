/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// set_lighting's key table and the value check of the lighting settings and parameters (shared by the MCP tool, the
// Lighting tab and the environment file; round 3, nit 11: the file loader no longer depends on the app's tool code)

#include "saiga/engine/lighting/LightingKeys.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
using Json = json11::Json;

std::string lightingRangeText(double lo, double hi)
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
bool lightingNumber(const Json& j, const std::string& name, double lo, double hi, double& out, std::string& error)
{
    if (!j.is_number() || !std::isfinite(j.number_value()) || j.number_value() < lo || j.number_value() > hi)
    {
        error = name + ": expected a finite number in " + lightingRangeText(lo, hi);
        return false;
    }
    out = j.number_value();
    return true;
}
const char* toneMapName(Vulkan::ToneMap t)
{
    return t == Vulkan::ToneMap::Aces ? "aces" : "identity";
}
const char* exposureModeName(Vulkan::ExposureMode m)
{
    return m == Vulkan::ExposureMode::Fixed ? "fixed" : "time_of_day";
}

namespace
{
Json vec3Json(const vec3& v)
{
    return Json::array{double(v(0)), double(v(1)), double(v(2))};
}
// The tool's helper names (the keys were written against them)
std::string rangeText(double lo, double hi)
{
    return lightingRangeText(lo, hi);
}
bool number(const Json& j, const std::string& name, double lo, double hi, double& out, std::string& error)
{
    return lightingNumber(j, name, lo, hi, out, error);
}

// ---- set_lighting's keys: each reads its value from the settings and writes a validated value into scratch copies

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

}  // namespace

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
        // (round 2, finding 14: the input of the fixed mode, not the frame's EV100 of get_lighting.exposure)
        k.back().description += " (exposure_mode fixed: the EV100; no effect in time_of_day mode)";
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

}  // namespace Engine
}  // namespace Saiga
