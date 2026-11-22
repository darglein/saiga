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
#include "saiga/export.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
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
    std::vector<vec2> curve = {vec2(-18, -1.5f), vec2(-10, -1),   vec2(-6, 2.5f),  vec2(-3, 7), vec2(0, 10.5f),
                               vec2(5, 12.3f),   vec2(10, 13.3f), vec2(30, 14.5f), vec2(90, 15)};
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

// ---- the debug views, overlays and previews (A8 of the S36 plan; RenderScene::setLightingDebug, LightingDebug)

// The scene debug view (one at a time; uniform branches of lighting.glsl). While a view other
// than None is active the frame latches ToneMap::Identity (no pipeline change). Data views write raw values into the
// fp16 target (Albedo, Normals as n * 0.5 + 0.5, Roughness, Metalness, Material = (roughness, metalness, ao), the
// factors); LightingOnly (white albedo, roughness 0.5), the *Only radiance views and the Exposure heat map write
// display colours (the frame's exposure and the ACES curve in the physical path, as clay in clay mode). The probe
// frames of probe_lighting write the radiance views raw (linear cd/m^2)
enum class LightingDebugView : uint8_t
{
    None,
    LightingOnly,
    Albedo,
    Normals,
    Roughness,
    Metalness,
    Material,
    DirectOnly,
    AmbientOnly,
    SpecularOnly,
    CloudShadow,
    ShadowFactor,
    Exposure,
    Count
};
// snake_case names (set_lighting_debug, screenshot lighting_view)
inline const char* lightingDebugViewName(LightingDebugView v)
{
    static const char* names[] = {"none",         "lighting_only", "albedo",      "normals",      "roughness",
                                  "metalness",    "material",      "direct_only", "ambient_only", "specular_only",
                                  "cloud_shadow", "shadow_factor", "exposure"};
    static_assert(sizeof(names) / sizeof(names[0]) == size_t(LightingDebugView::Count), "lightingDebugViewName");
    return v < LightingDebugView::Count ? names[size_t(v)] : "none";
}
inline std::optional<LightingDebugView> lightingDebugViewFromName(const std::string& name)
{
    for (int i = 0; i < int(LightingDebugView::Count); ++i)
        if (name == lightingDebugViewName(LightingDebugView(i))) return LightingDebugView(i);
    return std::nullopt;
}

// The overlays (flags, combinable), drawn on top in the scene pass by DebugLines. a1: SunDirection (lines from the
// camera's focus towards the sun and the moon); a2 and b add theirs
enum LightingOverlay : uint32_t
{
    OverlaySunDirection = 1u << 0,
    OverlayAll          = OverlaySunDirection
};
inline const std::vector<std::pair<const char*, uint32_t>>& lightingOverlayNames()
{
    static const std::vector<std::pair<const char*, uint32_t>> names = {{"sun_direction", OverlaySunDirection}};
    return names;
}

struct LightingDebugState
{
    LightingDebugView view = LightingDebugView::None;
    uint32_t overlays      = 0;
    bool operator==(const LightingDebugState& o) const { return view == o.view && overlays == o.overlays; }
    bool operator!=(const LightingDebugState& o) const { return !(*this == o); }
};

// The images LightingDebug converts into displayable RGBA8 previews and reduces (min, max, mean, histogram) on demand:
// a1b Luminance (the resolved HDR luminance in cd/m^2, or the display luminance in clay mode) and BrdfLut (capture
// only); a1c the atmosphere's TransmittanceLut and SkyViewLut (luminance, cd/m^2), the environment map's Environment
// (the front cube's mip as a cross, cd/m^2), Irradiance (its SH per direction, lux; capture only), the cloud layer's
// CloudCoverage (the coverage at the cloudiness over the tile) and CloudShadow (the cloud shadow factor on the plane of
// the camera height around the camera); a2 and b add theirs
enum class LightingImage : uint8_t
{
    Luminance,
    BrdfLut,
    TransmittanceLut,
    SkyViewLut,
    Environment,
    Irradiance,
    CloudCoverage,
    CloudShadow,
    Count
};
inline const char* lightingImageName(LightingImage i)
{
    static const char* names[] = {"luminance",   "brdf_lut",   "transmittance_lut", "sky_view_lut",
                                  "environment", "irradiance", "cloud_coverage",    "cloud_shadow"};
    static_assert(sizeof(names) / sizeof(names[0]) == size_t(LightingImage::Count), "lightingImageName");
    return i < LightingImage::Count ? names[size_t(i)] : "";
}
inline std::optional<LightingImage> lightingImageFromName(const std::string& name)
{
    for (int i = 0; i < int(LightingImage::Count); ++i)
        if (name == lightingImageName(LightingImage(i))) return LightingImage(i);
    return std::nullopt;
}

enum class ColorMap : uint8_t
{
    Gray,
    Turbo
};

// The value mapping of a preview: the range (auto: the GPU min / max of the source values; else [min, max]), log2 of
// the values before the mapping, the colour map. The histogram has 16 bins over the mapped range
struct PreviewMapping
{
    bool autoRange    = true;
    float min         = 0;
    float max         = 1;
    bool log          = false;
    ColorMap colorMap = ColorMap::Gray;
};

// The reduction of a preview's source values (after the log2 of the mapping, if any): over the finite values of the
// image's first channel (luminance: the luminance; brdf_lut: A)
struct PreviewStatistics
{
    float min = 0, max = 0, mean = 0;
    // The range the histogram and the preview used
    float rangeMin = 0, rangeMax = 1;
    std::array<uint32_t, 16> histogram = {};
    uint32_t count                     = 0;
};

// The state of the environment map (EnvironmentMap, S36a1c)
struct EnvironmentMapStatistics
{
    // The frame in which the front cube was completed (0: none yet) and the captures so far
    uint64_t lastCompleteFrame = 0;
    uint64_t captures          = 0;
    // The capture in progress: the steps done of the steps of one capture (capture, one per mip, the SH); 0 none
    int progress = 0, steps = 0;
    bool valid = false;
};

// The lighting's statistics of the newest recorded frame (RenderScene::lightingStatistics; a2 and b add theirs)
struct LightingStatistics
{
    uint64_t frame = 0;
    // 0 none, 1 the sun, 2 the moon
    int shadowLight    = 0;
    float sunElevation = 0, moonElevation = 0;  // degrees
    float ev100 = 0, exposure = 1, preExposure = 1;
    bool physical = false;
    LightingDebugState debug;
    // The ambient's SH9 (lux) and its dominant colour
    std::array<vec3, 9> sh = {};
    // The preview conversions recorded in the frame (0 without a request)
    uint32_t previewDispatches = 0;
    // The last luminance reduction (when requested) and its frame
    PreviewStatistics luminance;
    uint64_t luminanceFrame = 0;
    // The lighting's GPU memory (block, SH, BRDF LUT, dummies, previews, debug lines, the atmosphere's LUTs, the
    // environment map)
    uint64_t memoryBytes = 0;
    // S36a1c: the environment map and the atmosphere's transmittance / multiple-scattering recomputations
    EnvironmentMapStatistics environmentMap;
    uint64_t lutComputations = 0;
    // The cloud layer: drawn (and its shadows on) this frame, the mean coverage at the cloudiness
    bool clouds             = false;
    float cloudMeanCoverage = 0;
};

// The answer of a probe (RenderScene::requestProbe, probe_lighting): a short frame sequence at one pixel. Frame 1 is
// normal (the colour and the depth), then the data views Albedo, Normals, Material, the radiance views DirectOnly,
// AmbientOnly, SpecularOnly (raw) and CloudShadow (S36a1c); each copies the resolved fp16 pixel (the MSAA average). The
// camera and the environment are held during the sequence; consistent: the depth changed by at most 1% between the
// frames
struct LightingProbe
{
    ivec2 pixel  = ivec2(0, 0);
    bool inImage = false;
    // A surface was hit (the depth is not the far plane)
    bool hit        = false;
    int frames      = 0;
    bool consistent = true;
    bool physical   = false;
    float depth     = 0;  // reversed-Z
    Vec3 position   = Vec3::Zero();
    double distance = 0;
    // Frame 1: the radiance (cd/m^2; clay: the display value), its luminance and EV100 (log2(L * 100 / 12.5))
    vec3 color      = vec3(0, 0, 0);
    float luminance = 0, ev100 = 0;
    vec3 albedo     = vec3(0, 0, 0);
    vec3 normal     = vec3(0, 0, 0);
    float roughness = 0, metalness = 0, ao = 0;
    vec3 direct = vec3(0, 0, 0), ambient = vec3(0, 0, 0), specular = vec3(0, 0, 0);
    // The shadow factors (a1: the cloud factor, the CloudShadow frame; a2 adds the cascades and the horizon)
    float cloudShadow = 1, shadowFactor = 1;
};

// The fixed light of clay mode (A3 of the S36 plan; D7): independent of the time of day and the weather, defined so
// that clay reproduces the S19a light model exactly. Used by shadeSurface in clay mode (ToneMap::Identity)
// and in a scene without an environment (RenderScene::setEnvironment never called)
struct StudioEnvironment
{
    vec3 sunDir        = vec3(0.3113996f, 0.7784989f, 0.5449493f);  // towards the sun (normalized)
    float sunIntensity = 0.8f;
    float ambient      = 0.35f;
    // The hemisphere ambient ambient * (0.6 + 0.4 n.y) (else constant)
    bool hemisphere = false;
    // The clay albedo of the prototype's surfaces (terrain, water, grass, foliage); flat-colour instances keep their
    // material colour
    float albedo = 0.55f;
    // The height fog (protoApplyFog of S19a): density, height falloff (1/m), colour; its base is the sea level
    bool fog         = false;
    float fogDensity = 0.00015f, fogFalloff = 0.0012f;
    vec3 fogColor = vec3(0.42f, 0.42f, 0.42f);
};

// S18's fixed light: a RenderScene without the prototype environment
inline StudioEnvironment sceneStudio()
{
    return StudioEnvironment();
}
// S19a's PrototypeSettings defaults (sun azimuth 35, elevation 38, intensity 1, hemisphere ambient 0.25, albedo 0.55,
// fog 0.00015 / 0.0012, gray fog 0.42): the scene with the foliage texture (setFoliageTexture), with
// PrototypeSettings::albedo
SAIGA_VULKAN_API StudioEnvironment prototypeStudio();

}  // namespace Vulkan
}  // namespace Saiga
