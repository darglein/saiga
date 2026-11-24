/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The celestial model (sun and moon positions, the shadow light).

#include "saiga/engine/lighting/Celestial.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr double kPi  = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

// The direction of a body with declination decl and hour angle h at latitude lat (radians)
Vec3 bodyDirection(double lat, double decl, double h)
{
    const double east  = -std::cos(decl) * std::sin(h);
    const double north = std::cos(lat) * std::sin(decl) - std::sin(lat) * std::cos(decl) * std::cos(h);
    const double up    = std::sin(lat) * std::sin(decl) + std::cos(lat) * std::cos(decl) * std::cos(h);
    Vec3 d(east, up, -north);
    const double n = d.norm();
    return n > 0 ? Vec3(d / n) : Vec3(0, 1, 0);
}
}  // namespace

double elevationDeg(const Vec3& dir)
{
    const double n = dir.norm();
    if (!(n > 0)) return 0;
    return std::asin(std::clamp(dir(1) / n, -1.0, 1.0)) / kDeg;
}

double azimuthDeg(const Vec3& dir)
{
    // Clockwise from north (-z) towards east (+x)
    const double a = std::atan2(dir(0), -dir(2)) / kDeg;
    return a < 0 ? a + 360 : a;
}

Vec3 directionFromAngles(double azimuth, double elevation)
{
    const double az = azimuth * kDeg, el = elevation * kDeg;
    return Vec3(std::cos(el) * std::sin(az), std::sin(el), -std::cos(el) * std::cos(az));
}

CelestialState celestialState(const CelestialConfig& c, double timeOfDay, int64_t day)
{
    CelestialState s;
    const double lat       = c.latitudeDeg * kDeg;
    const double year      = c.yearLengthDays > 0 ? c.yearLengthDays : 365;
    const double dayOfYear = c.dayOfYearAtDay0 + double(day) + (timeOfDay - 12.0) / 24.0;
    // 0 at the spring equinox (day of the year 80, at noon)
    const double decl = c.axialTiltDeg * kDeg * std::sin(2 * kPi * (dayOfYear - 80) / year);
    const double h    = (timeOfDay - 12.0) / 24.0 * 2 * kPi;
    s.hourAngle       = h;
    s.sunDir          = bodyDirection(lat, decl, h);
    s.moonPhase       = c.moonPhase - std::floor(c.moonPhase);
    // The moon: the hour angle lags the sun's by the phase; the declination mirrors the sun's towards the full moon
    const double phaseAngle = 2 * kPi * s.moonPhase;
    s.moonDir               = bodyDirection(lat, decl * std::cos(phaseAngle), h - phaseAngle);
    s.sunIlluminance        = c.sunIlluminance;
    // The lit fraction of the disc: 0 at new moon, 1 at full moon
    s.moonIlluminance  = c.moonIlluminance * 0.5 * (1 - std::cos(phaseAngle));
    s.sunElevationDeg  = elevationDeg(s.sunDir);
    s.moonElevationDeg = elevationDeg(s.moonDir);
    s.sunAzimuthDeg    = azimuthDeg(s.sunDir);
    s.moonAzimuthDeg   = azimuthDeg(s.moonDir);
    return s;
}

Mat3 starRotation(const CelestialConfig& c, double hourAngle)
{
    // Rodrigues: a rotation by -hourAngle about the pole (the sky turns from east to west)
    const double lat = c.latitudeDeg * kDeg;
    const Vec3 k(0, std::sin(lat), -std::cos(lat));
    const double a = -hourAngle, ca = std::cos(a), sa = std::sin(a);
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r(i, j) = (1 - ca) * k(i) * k(j) + (i == j ? ca : 0.0);
    r(0, 1) -= sa * k(2);
    r(0, 2) += sa * k(1);
    r(1, 0) += sa * k(2);
    r(1, 2) -= sa * k(0);
    r(2, 0) -= sa * k(1);
    r(2, 1) += sa * k(0);
    return r;
}

double groundIlluminance(double top, double elevation)
{
    return elevation > 0 ? top * std::sin(elevation * kDeg) : 0.0;
}

ShadowLight selectShadowLight(const CelestialState& s, ShadowLight previous)
{
    const double sun  = groundIlluminance(s.sunIlluminance, s.sunElevationDeg);
    const double moon = groundIlluminance(s.moonIlluminance, s.moonElevationDeg);
    if (!(sun > 0) && !(moon > 0)) return ShadowLight::None;
    // Hysteresis factor 2: the other body takes over only when it is twice as bright
    if (previous == ShadowLight::Sun && sun > 0 && moon < 2 * sun) return ShadowLight::Sun;
    if (previous == ShadowLight::Moon && moon > 0 && sun < 2 * moon) return ShadowLight::Moon;
    return sun >= moon ? ShadowLight::Sun : ShadowLight::Moon;
}

const char* shadowLightName(ShadowLight light)
{
    switch (light)
    {
        case ShadowLight::Sun:
            return "sun";
        case ShadowLight::Moon:
            return "moon";
        default:
            return "none";
    }
}

}  // namespace Engine
}  // namespace Saiga
