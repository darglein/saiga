/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The environment extract: reads WorldTime and the space's Weather, writes nothing.

#include "saiga/engine/lighting/EnvironmentExtract.h"

#include "saiga/engine/lighting/EnvironmentWrites.h"
#include "saiga/engine/world/World.h"


namespace Saiga
{
namespace Engine
{
namespace
{
constexpr double kTwoPi = 6.28318530717958647692;
// The angular frequencies (rad/s) of the phases: the trees' wind (3), the grass's (2), the cloud drift remainder (a
// period of 64 s of the drift), the flicker base
constexpr double kPhaseRates[7] = {0.9, 1.7, 3.1, 1.3, 2.3, kTwoPi / 64, 10.0};

vec3 toFloat(const Vec3& v)
{
    return vec3(float(v(0)), float(v(1)), float(v(2)));
}
}  // namespace

EnvironmentSample sampleEnvironment(const World& world, SpaceId space, double alpha,
                                    const Vulkan::EnvironmentSettings& settings, const EnvironmentSetup& setup)
{
    EnvironmentSample out;
    const double tr = world.time() + (alpha - 1) * world.dt();
    out.renderTime  = tr;

    // The time of day at the render time
    WorldTime t;
    out.hasTime = hasWorldTime(world);
    if (out.hasTime)
    {
        t = world.readGlobal<WorldTime>();
        normaliseWorldTime(t);
        t.timeOfDay += t.rate * (tr - t.validAt) / 3600.0;
        normaliseWorldTime(t);
        t.validAt = tr;
    }
    out.time = t;

    // The space's weather
    Weather w;
    if (hasWeather(world) && space < world.spaceCount() && world.alive(world.spaceStateEntity(space)))
    {
        w = world.read<Weather>(world.spaceStateEntity(space));
        normaliseWeather(w);
        w.cloudOffset += w.wind * ((tr - w.validAt) * setup.cloudWindScale);
        normaliseWeather(w);
        w.validAt = tr;
    }
    out.weather = w;

    // Sun, moon and stars
    CelestialState c = celestialState(setup.celestial, t.timeOfDay, t.day);
    if (settings.sunOverride)
    {
        c.sunDir          = directionFromAngles(settings.sunOverride->azimuthDeg, settings.sunOverride->elevationDeg);
        c.sunElevationDeg = elevationDeg(c.sunDir);
        c.sunAzimuthDeg   = azimuthDeg(c.sunDir);
    }
    out.celestial   = c;
    out.shadowLight = selectShadowLight(c);
    out.enclosed    = space < world.spaceCount() && world.spaceTemplate(space).kind == SpaceKind::Enclosed;

    Vulkan::EnvironmentState& s = out.state;
    s.sunDir                    = toFloat(c.sunDir);
    s.sunIlluminance            = float(c.sunIlluminance);
    s.moonDir                   = toFloat(c.moonDir);
    s.moonIlluminance           = float(c.moonIlluminance);
    s.moonPhase                 = float(c.moonPhase);
    s.starIntensity             = float(setup.celestial.starIntensity * double(settings.starIntensity));
    const Mat3 stars            = starRotation(setup.celestial, c.hourAngle);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) s.starRotation(i, j) = float(stars(i, j));
    s.cloudiness  = float(w.cloudiness);
    s.rain        = float(w.rain);
    s.fogDensity  = float(w.fogDensity);
    s.snowCover   = float(w.snowCover);
    s.wind        = vec2(float(w.wind(0)), float(w.wind(1)));
    s.cloudOffset = vec2(float(w.cloudOffset(0)), float(w.cloudOffset(1)));

    // Exposure (A3)
    const Vulkan::ExposureSettings& e = settings.exposure;
    out.curveEv100                    = e.curveEv100(float(c.sunElevationDeg));
    s.ev100                           = e.ev100At(float(c.sunElevationDeg));
    s.exposureSource =
        e.mode == Vulkan::ExposureMode::Fixed ? Vulkan::ExposureSource::Fixed : Vulkan::ExposureSource::Curve;
    s.ambientScale = settings.ambient.scale;
    s.minAmbient   = settings.ambient.minAmbient;
    s.clouds       = settings.clouds;
    s.atmosphere   = settings.atmosphere;

    if (out.enclosed)
    {
        // The template is the single authority of enclosed lighting: no sun, moon, sky or sky ambient
        const SpaceTemplate& tpl = world.spaceTemplate(space);
        s.enclosed               = true;
        s.enclosedAmbient        = toFloat(tpl.ambient);
        s.sunIlluminance         = 0;
        s.moonIlluminance        = 0;
        s.starIntensity          = 0;
        s.ev100                  = float(tpl.ev100) - e.compensation;
        s.exposureSource         = Vulkan::ExposureSource::Enclosed;
        out.shadowLight          = ShadowLight::None;
    }

    // Phases in double (no absolute time in a shader)
    for (size_t i = 0; i < 7; ++i) s.phases[i] = float(wrapPeriod(tr * kPhaseRates[i], kTwoPi));
    return out;
}

Vulkan::EnvironmentState environmentState(const World& world, SpaceId space, double alpha,
                                          const Vulkan::EnvironmentSettings& settings, const EnvironmentSetup& setup)
{
    return sampleEnvironment(world, space, alpha, settings, setup).state;
}

}  // namespace Engine
}  // namespace Saiga
