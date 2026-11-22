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
// The fog density multiplier of rain: the renderer's density is the weather's times (1 + kRainFogFactor * rain)
constexpr float kRainFogFactor = 4.0f;

// The largest cloud seed the Lighting tab and set_lighting take (an int)
constexpr int kMaxCloudSeed = 2147483647;

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
// set_lighting key. Part b of S36 adds the local light fields.
struct LightingParameters
{
    int environmentResolution      = 128;    // 32..512, a power of two
    float environmentUpdateDegrees = 0.5f;   // the shadow light's movement that re-captures the environment map
    bool clouds                    = true;   // the cloud layer and its shadows
    bool forceFullUpdates          = false;  // golden mode (§17.6): every incremental effect updates fully every frame
    // S36a2: the sun and moon cascades (A5 of the S36 plan)
    bool sunShadows                    = true;          // the cascaded shadow maps of the shadow light
    int cascadeCount                   = 4;             // 1..4
    int cascadeResolution              = 2048;          // 256..4096, a power of two
    float shadowDistance               = 1500;          // m, the far end of the last cascade
    float splitLambda                  = 0.8f;          // the log / linear blend of the splits
    std::array<int, 4> cascadeInterval = {1, 1, 1, 2};  // frames between the updates of each cascade (the first 1)
    float depthBiasTexels       = 1.0f;  // the receiver's constant bias along the light, in texels of its cascade
    float slopeBias             = 1.5f;  // the rasterizer's depthBiasSlopeFactor
    float normalOffsetTexels    = 1.0f;  // the receiver's offset along its normal, in texels of its cascade
    float cascadeBlend          = 0.1f;  // the blend band at a cascade's border, a fraction of its half extent
    int shadowLodBias           = 1;     // LOD levels added for cascades 2-3 (cascades 0-1: the main view's LOD)
    float smallCasterTexels     = 1.5f;  // casters whose radius projects below this many texels are skipped
    float maxCasterHeight       = 120;   // m above the receivers: the instances' cap towards the light
    float minShadowElevationDeg = 5;     // the elevation the instances' cap is built for at most
    float cascadeCasterDistance = 1371;  // m towards the light (horizontally): terrain casters in the cascades
    bool terrainCastsInCascades = true;  // the terrain is drawn into the cascades
    // S36a2: the far terrain horizon map (A6)
    bool horizonShadows     = true;   // the horizon map of the shadow light
    int horizonResolution   = 1024;   // 128..4096, a power of two
    float horizonExtent     = 32768;  // m, the square around the anchor
    int horizonRowsPerFrame = 64;     // the base of the incremental update (more when the light moves fast)
    // S36b: the local lights (B2)
    bool localLights     = false;  // the point and spot lights of the scene
    bool clusteredLights = true;   // false: every fragment loops over every light (the correctness reference)
    int maxLights        = 1024;   // the lights of a frame (1..4096), the most important ones (the rest: over budget)
    // S36b: the cluster grid (B3): screen tiles x, y and depth slices z (slice 0 to clusterNear, then exponential to
    // clusterFar, beyond it no local light), the fixed slots per cluster
    int clusterX            = 16;    // 1..64
    int clusterY            = 9;     // 1..64
    int clusterZ            = 24;    // 2..128
    float clusterNear       = 2;     // m
    float clusterFar        = 1000;  // m
    int maxLightsPerCluster = 128;   // 1..1024
    // S36b: the cached shadow atlas (B4): two D16 layers of atlasResolution^2 in tiles of atlasTile texels (a spot
    // light 1 tile, a point light 6), at most facesPerFrame faces rendered per frame (each a culling view: <= 11), the
    // receivers' bias in D16 steps and their normal offset in texels
    int atlasResolution     = 4096;  // 512..8192, a power of two
    int atlasTile           = 512;   // 64..atlasResolution, divides it
    int facesPerFrame       = 8;     // 0..11
    float atlasDepthBias    = 2;     // D16 steps (range / 65535)
    float atlasNormalOffset = 1;     // texels
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
    // S36a2: the cascade factor tinted by the cascade index (red, green, blue, yellow; gray outside every cascade)
    ShadowCascades,
    // S36a2: the horizon map's factor (raw)
    HorizonShadow,
    // S36b: the local lights' direct light alone (display colours)
    LocalLightsOnly,
    // S36b: the lights of the fragment's cluster as a heat map (turbo; an overflowed cluster magenta)
    ClusterLights,
    // S36b: the atlas shadow of the local lights (the highlighted light's, else the darkest of the shadowed lights that
    // reach the fragment; blue where none reaches)
    AtlasShadow,
    Count
};
// snake_case names (set_lighting_debug, screenshot lighting_view)
inline const char* lightingDebugViewName(LightingDebugView v)
{
    static const char* names[] = {
        "none",     "lighting_only",   "albedo",         "normals",           "roughness",      "metalness",
        "material", "direct_only",     "ambient_only",   "specular_only",     "cloud_shadow",   "shadow_factor",
        "exposure", "shadow_cascades", "horizon_shadow", "local_lights_only", "cluster_lights", "atlas_shadow"};
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
// camera's focus towards the sun and the moon); a2: CascadeColors (the surfaces tinted by their cascade, also with
// other views; in the shaders), CascadeBounds (each cascade's box and caster volume as a wireframe in its colour),
// CameraFrustumSplits (the split planes of the camera frustum the cascades were fitted to; with frozen cascades the
// frozen camera's); b adds theirs
enum LightingOverlay : uint32_t
{
    OverlaySunDirection        = 1u << 0,
    OverlayCascadeColors       = 1u << 1,
    OverlayCascadeBounds       = 1u << 2,
    OverlayCameraFrustumSplits = 1u << 3,
    // The horizon map's extent and its inner half (the anchor moves when the camera leaves it)
    OverlayHorizonAnchor = 1u << 4,
    // S36b: the volumes of the frame's lights (spheres of point lights, cones of spot lights, in the light's colour;
    // shadowed ones thick; the highlighted light white)
    OverlayLightVolumes = 1u << 5,
    // S36b: the cluster grid's depth slices and tiles (of the frozen camera with freeze cascades)
    OverlayClusterGrid = 1u << 6,
    // S36b: the owner (light handle and face) of each tile in the atlas preview of the Lighting tab
    OverlayAtlasOwners = 1u << 7,
    OverlayAll = OverlaySunDirection | OverlayCascadeColors | OverlayCascadeBounds | OverlayCameraFrustumSplits |
                 OverlayHorizonAnchor | OverlayLightVolumes | OverlayClusterGrid | OverlayAtlasOwners
};
inline const std::vector<std::pair<const char*, uint32_t>>& lightingOverlayNames()
{
    static const std::vector<std::pair<const char*, uint32_t>> names = {
        {"sun_direction", OverlaySunDirection},   {"cascade_colors", OverlayCascadeColors},
        {"cascade_bounds", OverlayCascadeBounds}, {"camera_frustum_splits", OverlayCameraFrustumSplits},
        {"horizon_anchor", OverlayHorizonAnchor}, {"light_volumes", OverlayLightVolumes},
        {"cluster_grid", OverlayClusterGrid},     {"atlas_owners", OverlayAtlasOwners}};
    return names;
}
// The display colours of the cascades (the ShadowCascades view, the overlays)
inline vec3 cascadeColor(int cascade)
{
    static const vec3 colors[4] = {vec3(1.0f, 0.25f, 0.2f), vec3(0.3f, 1.0f, 0.3f), vec3(0.3f, 0.5f, 1.0f),
                                   vec3(1.0f, 0.9f, 0.2f)};
    return cascade >= 0 && cascade < 4 ? colors[cascade] : vec3(0.6f, 0.6f, 0.6f);
}

struct LightingDebugState
{
    LightingDebugView view = LightingDebugView::None;
    uint32_t overlays      = 0;
    // S36b: the highlighted light (its handle ID; 0: none): white in LightVolumes, its row in the tab
    uint32_t highlightLight = 0;
    bool operator==(const LightingDebugState& o) const
    {
        return view == o.view && overlays == o.overlays && highlightLight == o.highlightLight;
    }
    bool operator!=(const LightingDebugState& o) const { return !(*this == o); }
};

// The images LightingDebug converts into displayable RGBA8 previews and reduces (min, max, mean, histogram) on demand:
// a1b Luminance (the resolved HDR luminance in cd/m^2, or the display luminance in clay mode) and BrdfLut (capture
// only); a1c the atmosphere's TransmittanceLut and SkyViewLut (luminance, cd/m^2), the environment map's Environment
// (the front cube's mip as a cross, cd/m^2), Irradiance (its SH per direction, lux; capture only), the cloud layer's
// CloudCoverage (the coverage at the cloudiness over the tile) and CloudShadow (the cloud shadow factor on the plane of
// the camera height around the camera), the a1c review the MultiScatteringLut (luminance); a2 the cascades' depth
// (Cascade: one layer, RenderScene's preview cascade; CascadeStrip: every layer side by side; the light-space depth 0
// at the near end, 1 at the far end, at most 1024 texels per layer, the minimum over the texels a preview texel
// covers), Horizon (the front horizon map's R or G shadow height, m relative to its reference); b adds theirs
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
    // S36a1c review: the atmosphere's multiple-scattering LUT (the luminance of rgb)
    MultiScatteringLut,
    Cascade,
    CascadeStrip,
    Horizon,
    // S36b: the lights per cluster (16 pixels per cluster; one slice or the maximum over the slices)
    Clusters,
    // S36b: the shadow atlas (a layer: static or dynamic; the linear depth, at most 1024 texels) and one tile (a
    // light's face at its full resolution)
    Atlas,
    AtlasTile,
    Count
};
inline const char* lightingImageName(LightingImage i)
{
    static const char* names[] = {"luminance",      "brdf_lut",      "transmittance_lut",
                                  "sky_view_lut",   "environment",   "irradiance",
                                  "cloud_coverage", "cloud_shadow",  "multi_scattering_lut",
                                  "cascade",        "cascade_strip", "horizon",
                                  "clusters",       "atlas",         "atlas_tile"};
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
    // The source's mip (the environment map's cube; 0 for the other images): part of the request, so a capture and the
    // tab's preview of different mips do not overwrite each other (each frame converts one)
    int mip = 0;
    // The source's layer or channel the same way (S36a2 review, finding 7): the cascade (Cascade) or the horizon map's
    // channel (Horizon: 0 R, 1 G); 0 for the other images
    int layer = 0;
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

// One shadow cascade (S36a2, LightingStatistics::cascades): its slice of the camera frustum, its light-space box (the
// centre in world coordinates, the half extent in xy, the shared depth range), texel size, update interval and the
// frame it was last rendered in, and the casters drawn into it the last time (instances and triangles from the GPU
// statistics; terrain nodes and triangles from the selection)
struct CascadeStatistics
{
    bool valid      = false;
    float splitNear = 0, splitFar = 0;   // m from the camera
    Vec3 center         = Vec3::Zero();  // world (the box's centre at mid depth)
    float halfExtent    = 0;             // m, the rectangle's half side
    float depthRange    = 0;             // m, the shared light-space depth range
    float texelSize     = 0;             // m
    int interval        = 1;
    uint64_t lastUpdate = 0;
    // Rendered in this frame (a staggered cascade every interval frames, or again when its de-duplication no longer
    // holds; nit 21)
    bool updatedThisFrame = false;
    uint32_t instances = 0, triangles = 0, terrainNodes = 0, terrainTriangles = 0;
    // The terrain nodes skipped by the de-duplication (drawn by the previous cascade, with their subtrees)
    uint32_t terrainDeduplicated = 0;
};

// The horizon map (S36a2, LightingStatistics::horizon): the front map's transform (its corner in integer metres, its
// reference height in m, the light it was computed for), the back map in progress (rows done), the blend between the
// previous and the front map, the rows computed in the frame, the recomputes started and completed, the chain level
// the march uses at 10 km (far shadows can grow by one cell's height range of it)
struct HorizonStatistics
{
    bool on = false, valid = false, computing = false;
    int resolution = 0;
    double extent = 0, texel = 0;
    ivec2 corner     = ivec2(0, 0);
    double reference = 0;
    vec3 light       = vec3(0, 1, 0);
    int rowsDone = 0, rowsThisFrame = 0;
    float blend         = 1;
    uint64_t recomputes = 0, completed = 0, lastComplete = 0;
    // Maps in progress restarted by a terrain change (their rows would mix the old and the new terrain)
    uint64_t restarts    = 0;
    int levelAt10km      = 0;
    // The chain buffer and the maps; chainBytes: the chain's part
    uint64_t memoryBytes = 0;
    uint64_t chainBytes  = 0;
};

// The local lights of a frame (S36b, LightingStatistics::lights): the scene's lights (submitted), those of the
// camera's space within clusterFar + range and in the frustum (candidates), the frame's (visible: at most maxLights,
// the most important), the rest (over budget); shadowed: visible with castShadows; without tiles: shadowed but no
// atlas tile (S36b step 4)
struct LocalLightStatistics
{
    bool on            = false;
    bool clustered     = false;
    uint32_t submitted = 0, candidates = 0, visible = 0, overBudget = 0, shadowed = 0, withoutTiles = 0;
    uint64_t memoryBytes = 0;
    // The clusters (B3) of the newest completed frame: the most lights in a cluster, the mean over the clusters, the
    // clusters with more lights than slots and the lights they dropped, the frame they come from
    uint32_t clusterCount = 0, clusterMax = 0, overflowedClusters = 0;
    double clusterMean     = 0;
    uint64_t droppedLights = 0;
    uint64_t clusterFrame  = 0;
    uint64_t clusterMemory = 0;
};

// One tile of the shadow atlas (ShadowAtlas::tiles; the Lighting tab's owner overlay, capture_lighting): its owner (a
// light handle, 0 free) and face, the frame of its last static render (0: not yet valid), whether its dynamic layer has
// content
struct AtlasTileInfo
{
    uint32_t owner        = 0;
    int face              = 0;
    bool staticValid      = false;
    uint64_t staticFrame  = 0;
    bool dynamicContent   = false;
    uint64_t dynamicFrame = 0;
};

// The atlas's statistics of the newest recorded frame (LightingStatistics::atlas)
struct AtlasStatistics
{
    bool on        = false;
    int resolution = 0, tile = 0;
    uint32_t tilesTotal = 0, tilesUsed = 0;
    // The faces rendered this frame (static / dynamic layer), the dynamic tiles cleared, the valid static faces that
    // were not re-rendered (cache hits), the static faces still waiting (queue), the shadowed lights with tiles and
    // without (the tile budget)
    uint32_t staticRenders = 0, dynamicRenders = 0, dynamicClears = 0, hits = 0, queue = 0;
    uint32_t lightsWithTiles = 0, lightsWithoutTiles = 0;
    int facesPerFrame    = 0;
    uint64_t memoryBytes = 0;
};

// One light of the frame (RenderScene::frameLights; the Lighting tab, list_lights): its handle, the buffer index,
// distance to the camera, importance and (S36b steps 3-4) the clusters it touches and its atlas tiles
struct FrameLightInfo
{
    uint32_t handle   = 0;
    uint32_t index    = 0;
    double distance   = 0;
    double importance = 0;
    uint32_t clusters = 0;
    // The atlas tiles (0: unshadowed), the first tile and the frame of its last render (0: not yet)
    int tiles           = 0;
    int firstTile       = -1;
    uint64_t lastRender = 0;
    bool shadowReady    = false;
};

// The lighting's statistics of the newest recorded frame (RenderScene::lightingStatistics; b adds its own)
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
    // S36a2: the cascades (on: rendered this frame or kept), the shadow light's direction they were built for, frozen
    bool shadows                              = false;
    bool frozen                               = false;
    int cascadeCount                          = 0;
    vec3 shadowLightDir                       = vec3(0, 1, 0);
    std::array<CascadeStatistics, 4> cascades = {};
    // The cascades' depth images (bytes) and whether the comparison sampler filters linearly
    uint64_t shadowMemoryBytes = 0;
    bool linearShadowFilter    = false;
    HorizonStatistics horizon;
    // The horizon map the frame's previews converted (the front map after the frame's update; capture_lighting's
    // anchor and reference: CodeRabbit #85 round 2)
    HorizonStatistics horizonPreview;
    // S36b: the local lights and the shadow atlas
    LocalLightStatistics lights;
    AtlasStatistics atlas;
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
    // The shadow factors (a1: the cloud factor, the CloudShadow frame; a2: the cascades, the ShadowCascades frame: the
    // cascade's factor and its index, -1 outside every cascade)
    float cloudShadow = 1, shadowFactor = 1;
    float cascadeShadow = 1;
    int cascade         = -1;
    // The horizon map (the HorizonShadow frame): its factor and the shadow heights R and G above the receiver (m;
    // negative: below it, lit)
    float horizonShadow = 1, horizonR = 0, horizonG = 0;
    // S36b: the cluster of the pixel (-1: none: no lights, beyond clusterFar or no surface), its lights in
    // importance order (the CPU reference of the assignment: total before the cap) with each light's reflected
    // radiance at the pixel (cd/m^2, the BRDF with the probed material and normal; S36b step 4: times its atlas shadow)
    int cluster           = -1;
    uint32_t clusterTotal = 0;
    struct LightContribution
    {
        uint32_t handle = 0;
        double distance = 0;
        vec3 radiance   = vec3(0, 0, 0);
    };
    std::vector<LightContribution> lights;
    // S36b: the atlas shadow (the AtlasShadow frame: the darkest factor of the shadowed lights that reach the pixel, 1:
    // none) and how many shadowed lights reach it
    float atlasShadow     = 1;
    int atlasShadowLights = 0;
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
