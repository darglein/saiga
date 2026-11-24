/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The celestial model of the environment (docs/engine_design.md §10.7; S36): sun and moon directions and illuminance
// from the time of day and the day count, pure and double (R-MATH-1). World frame: y up, north = -z, east = +x.

#pragma once

#include "saiga/core/math/Types.h"
#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// Part of EnvironmentSetup (constant after the world exists; the simulation and the extract read it)
struct CelestialConfig
{
    double latitudeDeg     = 45;
    double axialTiltDeg    = 23.44;
    double dayOfYearAtDay0 = 80;  // the equinox
    double yearLengthDays  = 365;
    double sunIlluminance  = 120000;  // lux at the top of the atmosphere
    double moonIlluminance = 0.27;    // lux of the full moon
    // 0 new, 0.5 full: a parameter, no synodic model. The moon's hour angle is the sun's minus 2 pi * moonPhase: the
    // full moon (0.5) stands opposite the sun (it rises at sunset), the first quarter (0.25) rises about at noon, the
    // new moon (0) stands next to the sun and gives no light
    double moonPhase     = 0.5;
    double starIntensity = 1;
};

struct CelestialState
{
    Vec3 sunDir            = Vec3(0, 1, 0);  // towards the body (unit, y up, north = -z)
    Vec3 moonDir           = Vec3(0, -1, 0);
    double sunIlluminance  = 0;  // lux at the top of the atmosphere
    double moonIlluminance = 0;  // lux at the top of the atmosphere, scaled by the lit fraction of the phase
    double moonPhase       = 0.5;
    double sunElevationDeg = 90, moonElevationDeg = -90;
    double sunAzimuthDeg = 180, moonAzimuthDeg = 0;  // clockwise from north: 90 east, 180 south
    // The sun's hour angle (radians, 0 at noon); rotates the stars about the celestial pole
    double hourAngle = 0;
};

// The bodies of the time of day (hours, [0, 24)) on the given day: the solar declination from the day of the year, the
// hour angle from the time
SAIGA_ENGINE_API CelestialState celestialState(const CelestialConfig& config, double timeOfDay, int64_t day);

// Elevation and azimuth (degrees, azimuth clockwise from north = -z) of a direction, and the inverse
SAIGA_ENGINE_API double elevationDeg(const Vec3& dir);
SAIGA_ENGINE_API double azimuthDeg(const Vec3& dir);
SAIGA_ENGINE_API Vec3 directionFromAngles(double azimuthDeg, double elevationDeg);

// The rotation of the star sphere: the celestial frame rotated by the hour angle about the celestial pole (north,
// raised by the latitude)
SAIGA_ENGINE_API Mat3 starRotation(const CelestialConfig& config, double hourAngle);

// The body that casts the shadows and is the "shadow light" (D13): the one with the higher ground illuminance (top
// illuminance times the sine of the elevation, 0 below the horizon), with a hysteresis factor of 2 against the previous
// choice (no alternation); None when both are below the horizon or dark
enum class ShadowLight : uint8_t
{
    None,
    Sun,
    Moon
};
SAIGA_ENGINE_API double groundIlluminance(double topIlluminance, double elevationDeg);
SAIGA_ENGINE_API ShadowLight selectShadowLight(const CelestialState& state, ShadowLight previous = ShadowLight::None);
SAIGA_ENGINE_API const char* shadowLightName(ShadowLight light);

}  // namespace Engine
}  // namespace Saiga
