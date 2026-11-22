/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Reading data/<game>/environment.json: the JSON object through visit structs that mirror the setup, settings and
// lighting parameters (the render types in saiga_vulkan have no visit of their own).

#include "saiga/engine/lighting/EnvironmentFile.h"

#include "saiga/core/util/assert.h"
#include "saiga/engine/data/DataFile.h"
#include "saiga/engine/ecs/VisitJson.h"

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
    out                      = f;
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
