/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Reading data/<game>/environment.json: the JSON object through visit structs that mirror the setup, settings and
// lighting parameters (the render types in saiga_vulkan have no visit of their own).

#include "saiga/engine/lighting/EnvironmentFile.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/app/LightingTools.h"
#include "saiga/engine/data/DataFile.h"
#include "saiga/engine/ecs/VisitJson.h"

#include <cmath>

namespace Saiga
{
namespace Engine
{
namespace
{
struct CelestialDoc
{
    static constexpr const char* name = "celestial";
    static constexpr uint32_t version = 1;
    CelestialConfig c;
    template <class V>
    void visit(V& v)
    {
        v("latitudeDeg", c.latitudeDeg);
        v("axialTiltDeg", c.axialTiltDeg);
        v("dayOfYearAtDay0", c.dayOfYearAtDay0);
        v("yearLengthDays", c.yearLengthDays);
        v("sunIlluminance", c.sunIlluminance);
        v("moonIlluminance", c.moonIlluminance);
        v("moonPhase", c.moonPhase);
        v("starIntensity", c.starIntensity);
    }
};

struct ExposureDoc
{
    static constexpr const char* name        = "exposure";
    static constexpr uint32_t version        = 1;
    static constexpr const char* modeNames[] = {"timeOfDay", "fixed"};
    Vulkan::ExposureSettings e;
    template <class V>
    void visit(V& v)
    {
        v("mode", e.mode, EnumNames{modeNames});
        v("ev100", e.ev100);
        v("curve", e.curve);
        v("compensation", e.compensation);
    }
};

struct AmbientDoc
{
    static constexpr const char* name = "ambient";
    static constexpr uint32_t version = 1;
    Vulkan::AmbientSettings a;
    template <class V>
    void visit(V& v)
    {
        v("scale", a.scale);
        v("minAmbient", a.minAmbient);
    }
};

struct CloudsDoc
{
    static constexpr const char* name = "clouds";
    static constexpr uint32_t version = 1;
    Vulkan::CloudSettings c;
    double windScale = 1;
    template <class V>
    void visit(V& v)
    {
        v("altitude", c.altitude);
        v("radius", c.radius);
        v("seed", c.seed);
        v("shadowFade", c.shadowFade);
        v("fogFalloff", c.fogFalloff);
        v("windScale", windScale);
    }
};

struct AtmosphereDoc
{
    static constexpr const char* name = "atmosphere";
    static constexpr uint32_t version = 1;
    Vulkan::AtmosphereSettings a;
    float starIntensity = 1;
    template <class V>
    void visit(V& v)
    {
        v("rayleighScale", a.rayleighScale);
        v("mieScale", a.mieScale);
        v("ozoneScale", a.ozoneScale);
        v("groundAlbedo", a.groundAlbedo);
        v("starIntensity", starIntensity);
    }
};

struct LightingDoc
{
    static constexpr const char* name = "lighting";
    static constexpr uint32_t version = 1;
    Vulkan::LightingParameters p;
    template <class V>
    void visit(V& v)
    {
        v("environmentResolution", p.environmentResolution);
        v("environmentUpdateDegrees", p.environmentUpdateDegrees);
        v("clouds", p.clouds);
        v("forceFullUpdates", p.forceFullUpdates);
        // S36a2: the cascades
        v("sunShadows", p.sunShadows);
        v("cascadeCount", p.cascadeCount);
        v("cascadeResolution", p.cascadeResolution);
        v("shadowDistance", p.shadowDistance);
        v("splitLambda", p.splitLambda);
        v("cascadeInterval", p.cascadeInterval);
        v("depthBiasTexels", p.depthBiasTexels);
        v("slopeBias", p.slopeBias);
        v("normalOffsetTexels", p.normalOffsetTexels);
        v("cascadeBlend", p.cascadeBlend);
        v("shadowLodBias", p.shadowLodBias);
        v("smallCasterTexels", p.smallCasterTexels);
        v("maxCasterHeight", p.maxCasterHeight);
        v("minShadowElevationDeg", p.minShadowElevationDeg);
        v("cascadeCasterDistance", p.cascadeCasterDistance);
        v("terrainCastsInCascades", p.terrainCastsInCascades);
        // S36a2: the horizon map
        v("horizonShadows", p.horizonShadows);
        v("horizonResolution", p.horizonResolution);
        v("horizonExtent", p.horizonExtent);
        v("horizonRowsPerFrame", p.horizonRowsPerFrame);
        // S36b: the local lights, the clusters, the shadow atlas
        v("localLights", p.localLights);
        v("clusteredLights", p.clusteredLights);
        v("maxLights", p.maxLights);
        v("clusterX", p.clusterX);
        v("clusterY", p.clusterY);
        v("clusterZ", p.clusterZ);
        v("clusterNear", p.clusterNear);
        v("clusterFar", p.clusterFar);
        v("maxLightsPerCluster", p.maxLightsPerCluster);
        v("atlasResolution", p.atlasResolution);
        v("atlasTile", p.atlasTile);
        v("facesPerFrame", p.facesPerFrame);
        v("atlasDepthBias", p.atlasDepthBias);
        v("atlasNormalOffset", p.atlasNormalOffset);
    }
};

// The top-level object (fromJson visits it directly: no name or version)
struct EnvironmentDoc
{
    CelestialDoc celestial;
    ExposureDoc exposure;
    AmbientDoc ambient;
    CloudsDoc clouds;
    AtmosphereDoc atmosphere;
    LightingDoc lighting;
    template <class V>
    void visit(V& v)
    {
        v("celestial", celestial);
        v("exposure", exposure);
        v("ambient", ambient);
        v("clouds", clouds);
        v("atmosphere", atmosphere);
        v("lighting", lighting);
    }
};
// The values without a set_lighting key: finite and in range; the error names the key path
bool validateFileValues(const EnvironmentFile& f, std::string& error)
{
    std::string errors;
    auto check = [&errors](const char* key, double v, double lo, double hi)
    {
        if (std::isfinite(v) && v >= lo && v <= hi) return;
        errors += (errors.empty() ? "" : "; ") + std::string(key) + ": expected a finite number in [" +
                  std::to_string(lo) + ", " + std::to_string(hi) + "]";
    };
    const CelestialConfig& c = f.setup.celestial;
    check("celestial.latitudeDeg", c.latitudeDeg, -90, 90);
    check("celestial.axialTiltDeg", c.axialTiltDeg, -90, 90);
    check("celestial.dayOfYearAtDay0", c.dayOfYearAtDay0, -1e6, 1e6);
    check("celestial.yearLengthDays", c.yearLengthDays, 1, 1e6);
    check("celestial.sunIlluminance", c.sunIlluminance, 0, 1e7);
    check("celestial.moonIlluminance", c.moonIlluminance, 0, 1e7);
    check("celestial.moonPhase", c.moonPhase, 0, 1);
    check("celestial.starIntensity", c.starIntensity, 0, 1e6);
    check("clouds.windScale", f.setup.cloudWindScale, -100, 100);
    // (round 2, finding 9: the fog falloff and the atmosphere's scales have set_lighting keys, checked with their
    // ranges by validateLightingValues; the looser ranges here were dead)
    if (errors.empty()) return true;
    error = errors;
    return false;
}
}  // namespace

bool tryReadEnvironmentFile(const std::string& path, EnvironmentFile& out, std::string& error)
{
    json11::Json json;
    if (!readJsonFile(path, json, error)) return false;
    if (!json.is_object())
    {
        error = path + ": expected an object";
        return false;
    }
    EnvironmentDoc doc;
    std::string e;
    if (!fromJson(json, doc, &e, true))
    {
        error = path + ": " + e;
        return false;
    }
    EnvironmentFile f;
    f.setup.celestial        = doc.celestial.c;
    f.setup.cloudWindScale   = doc.clouds.windScale;
    f.settings.exposure      = doc.exposure.e;
    f.settings.ambient       = doc.ambient.a;
    f.settings.clouds        = doc.clouds.c;
    f.settings.atmosphere    = doc.atmosphere.a;
    f.settings.starIntensity = doc.atmosphere.starIntensity;
    f.lighting               = doc.lighting.p;
    // R-DATA-1 also for the values: the ranges of set_lighting (its key names) and of the values without a key
    if (!validateFileValues(f, e) || !validateLightingValues(f.settings, f.lighting, e))
    {
        error = path + ": " + e;
        return false;
    }
    out = f;
    return true;
}

EnvironmentFile readEnvironmentFile(const std::string& path)
{
    EnvironmentFile f;
    std::string error;
    if (!tryReadEnvironmentFile(path, f, error)) SAIGA_EXIT_ERROR("environment file: " + error);
    return f;
}

}  // namespace Engine
}  // namespace Saiga
