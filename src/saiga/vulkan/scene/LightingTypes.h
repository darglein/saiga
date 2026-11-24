/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The value types of the lighting (docs/engine_design.md §10.7; S36): the presentation settings of the environment
// (EnvironmentSettings: exposure, ambient, clouds, atmosphere, the sun override; owned mutably by the engine app), the
// renderer's lighting parameters (LightingParameters, RenderSceneParameters::lighting) and the exposure rule. Render
// data: float (R-MATH-1).

#pragma once

#include "saiga/core/math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// How the frame's EV100 is chosen: TimeOfDay reads ExposureSettings::curve at the sun elevation, Fixed uses ev100
enum class ExposureMode : uint8_t
{
    TimeOfDay,
    Fixed
};

// Where the EV100 of a frame came from (statistics): the curve, the fixed value, or an enclosed space's template
enum class ExposureSource : uint8_t
{
    Curve,
    Fixed,
    Enclosed
};

// The exposure of a scene with an environment, in EV100 (A3 of the S36 plan). The effective exposure of the frame is
// exposureFromEv100(ev100At(sunElevation), RenderSceneParameters::exposure).
struct ExposureSettings
{
    ExposureMode mode = ExposureMode::TimeOfDay;
    float ev100       = 15;  // Fixed
    // (sun elevation in degrees, EV100), ascending in the elevation; piecewise linear, clamped at the ends
    std::vector<vec2> curve = {vec2(-18, -1), vec2(-6, 3), vec2(0, 9), vec2(10, 13), vec2(30, 14.5f), vec2(90, 15)};
    // Subtracted from the EV100 (+1: one stop brighter, the same as an exposure multiplier of 2)
    float compensation = 0;

    // The curve at an elevation (without the compensation); 15 for an empty curve
    float curveEv100(float sunElevationDeg) const
    {
        if (curve.empty()) return 15;
        if (!(sunElevationDeg > curve.front()(0))) return curve.front()(1);
        if (sunElevationDeg >= curve.back()(0)) return curve.back()(1);
        for (size_t i = 1; i < curve.size(); ++i)
        {
            if (sunElevationDeg > curve[i](0)) continue;
            const vec2 a = curve[i - 1], b = curve[i];
            const float span = b(0) - a(0);
            const float t    = span > 0 ? (sunElevationDeg - a(0)) / span : 1.0f;
            return a(1) + (b(1) - a(1)) * t;
        }
        return curve.back()(1);
    }
    // The frame's EV100 at this sun elevation: the curve (TimeOfDay) or ev100 (Fixed), minus the compensation
    float ev100At(float sunElevationDeg) const
    {
        return (mode == ExposureMode::Fixed ? ev100 : curveEv100(sunElevationDeg)) - compensation;
    }
};

// The exposure rule of a scene with an environment (D18): the multiplier (RenderSceneParameters::exposure, --exposure)
// over the photometric exposure 1 / (1.2 * 2^ev100), clamped to [2^-24, 2^10]
inline float exposureFromEv100(float ev100, float multiplier = 1)
{
    const double e = double(multiplier) / (1.2 * std::exp2(double(ev100)));
    if (!(e == e)) return 1.0f;
    return float(std::clamp(e, std::exp2(-24.0), std::exp2(10.0)));
}

// The ambient light (A3): the sky ambient times scale, at least minAmbient (lux-equivalent, the night floor)
struct AmbientSettings
{
    float scale     = 1;
    vec3 minAmbient = vec3(0.002f, 0.002f, 0.002f);
};

// The 2D cloud layer (A4): altitude above sea level and radius of the mesh (m), the seed of the coverage texture, the
// distance over which cloud shadows fade to the mean coverage (m) and the height fog's falloff (1/m)
struct CloudSettings
{
    float altitude   = 2000;
    float radius     = 40000;
    uint32_t seed    = 1;
    float shadowFade = 8000;
    float fogFalloff = 0.0012f;
};

// The atmosphere's parameters (Hillaire 2020, Earth defaults; A4): scales of the Rayleigh, Mie and ozone coefficients
// and the albedo of the ground below the horizon in the environment map
struct AtmosphereSettings
{
    float rayleighScale = 1;
    float mieScale      = 1;
    float ozoneScale    = 1;
    float groundAlbedo  = 0.15f;
};

// Freezes the sun direction for inspection (the Lighting tab's "override sun", set_lighting sun_override)
struct SunOverride
{
    float azimuthDeg   = 180;  // clockwise from north (-z): 90 east (+x), 180 south (+z)
    float elevationDeg = 45;
};

// The presentation settings of the environment (finding 13, D16): owned mutably by the engine app
// (EngineApp::environmentSettings), changed at run time by the Lighting tab, set_lighting and the game; read by the
// environment extract every frame. Not world state (not replicated, not saved)
struct EnvironmentSettings
{
    ExposureSettings exposure;
    AmbientSettings ambient;
    CloudSettings clouds;
    AtmosphereSettings atmosphere;
    // A multiplier of the stars' brightness (CelestialConfig::starIntensity)
    float starIntensity = 1;
    std::optional<SunOverride> sunOverride;
};

// The renderer's lighting parameters (RenderSceneParameters::lighting, RenderScene::setLightingParameters; the target
// of S37's quality tiers; R-MECH-1). Every field is changeable at run time (applied at the next frame start) and has a
// set_lighting key. Later parts of S36 add the shadow (a2) and local light (b) fields.
struct LightingParameters
{
    int environmentResolution      = 128;    // 32..512, a power of two
    float environmentUpdateDegrees = 0.5f;   // the shadow light's movement that re-captures the environment map
    bool clouds                    = true;   // the cloud layer and its shadows
    bool forceFullUpdates          = false;  // golden mode (§17.6): every incremental effect updates fully every frame
};

}  // namespace Vulkan
}  // namespace Saiga
