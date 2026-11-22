/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The atmosphere of S36 (A4 of docs/stages/S36_lighting.plan.md; Hillaire 2020, "A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique"; Saiga::Vulkan::Atmosphere): the Earth's medium (Rayleigh, Mie, ozone) with the
// scales of AtmosphereSettings, the phase functions, the parametrizations of the transmittance LUT (Bruneton), the
// multiple-scattering LUT and the sky-view LUT, and the lookups the scene shaders use (the sun's transmittance at a
// receiver, the sky radiance in a direction). Distances inside the atmosphere in km, heights above the sea level; y up.
// The sky-view LUT and the environment cube store radiance (cd/m^2) times kSkyLutScale (the fp16 range of a sky under
// 120 000 lux).
#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

const float kAtmPi          = 3.14159265358979;
const float kGroundRadiusKm = 6360.0;
const float kTopRadiusKm    = 6460.0;
const float kSkyLutScale    = 1.0 / 16.0;

// Scattering and absorption coefficients at the sea level (1/km)
const vec3 kRayleighScattering = vec3(5.802, 13.558, 33.1) * 1e-3;
const float kMieScattering     = 3.996e-3;
const float kMieExtinction     = 4.40e-3;
const vec3 kOzoneAbsorption    = vec3(0.650, 1.881, 0.085) * 1e-3;

// The medium at a height (km) with the scales (x Rayleigh, y Mie, z ozone): the scattering of both and the extinction
void atmosphereMedium(float heightKm, vec3 scales, out vec3 rayleigh, out float mie, out vec3 extinction)
{
    float h    = max(heightKm, 0.0);
    float rd   = exp(-h / 8.0);
    float md   = exp(-h / 1.2);
    float od   = max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    rayleigh   = kRayleighScattering * rd * scales.x;
    mie        = kMieScattering * md * scales.y;
    extinction = rayleigh + vec3(kMieExtinction * md * scales.y) + kOzoneAbsorption * od * scales.z;
}

float rayleighPhase(float c)
{
    return 3.0 / (16.0 * kAtmPi) * (1.0 + c * c);
}

// Cornette-Shanks with g = 0.8
float miePhase(float c)
{
    const float g = 0.8;
    float k       = 3.0 / (8.0 * kAtmPi) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}

// The distance from radius r along the cosine mu to the sphere of radius R from inside (the far root); -1 if none
float distanceToSphere(float r, float mu, float R)
{
    float d = r * r * (mu * mu - 1.0) + R * R;
    return d < 0.0 ? -1.0 : -r * mu + sqrt(d);
}

bool rayHitsGround(float r, float mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + kGroundRadiusKm * kGroundRadiusKm >= 0.0;
}

float distanceToGround(float r, float mu)
{
    return -r * mu - sqrt(max(r * r * (mu * mu - 1.0) + kGroundRadiusKm * kGroundRadiusKm, 0.0));
}

// The transmittance LUT (Bruneton): (r, mu) <-> uv; x: the distance to the top between its minimum and maximum, y:
// the height
vec2 transmittanceUv(float r, float mu)
{
    float H   = sqrt(kTopRadiusKm * kTopRadiusKm - kGroundRadiusKm * kGroundRadiusKm);
    float rho = sqrt(max(r * r - kGroundRadiusKm * kGroundRadiusKm, 0.0));
    float d   = max(distanceToSphere(r, mu, kTopRadiusKm), 0.0);
    float dm  = kTopRadiusKm - r;
    float dM  = rho + H;
    return vec2((d - dm) / max(dM - dm, 1e-6), rho / H);
}

void transmittanceRMu(vec2 uv, out float r, out float mu)
{
    float H   = sqrt(kTopRadiusKm * kTopRadiusKm - kGroundRadiusKm * kGroundRadiusKm);
    float rho = H * uv.y;
    r         = sqrt(rho * rho + kGroundRadiusKm * kGroundRadiusKm);
    float dm  = kTopRadiusKm - r;
    float dM  = rho + H;
    float d   = dm + uv.x * (dM - dm);
    mu        = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu        = clamp(mu, -1.0, 1.0);
}

// The radius of a height above the sea level (m), inside the atmosphere
float atmosphereRadius(float heightM)
{
    return clamp(kGroundRadiusKm + heightM * 1e-3, kGroundRadiusKm + 1e-3, kTopRadiusKm - 1e-3);
}

// The transmittance from (r, mu) to the top of the atmosphere (no ground test)
vec3 transmittanceToTop(sampler2D lut, float r, float mu)
{
    return textureLod(lut, transmittanceUv(r, mu), 0.0).rgb;
}

// The transmittance of a body's light (direction l towards it) at the radius r: 0 behind the planet, a soft edge of
// about the sun disc's size at the horizon
vec3 bodyTransmittanceAtRadius(sampler2D lut, float r, vec3 l)
{
    float horizon = -sqrt(max(1.0 - kGroundRadiusKm * kGroundRadiusKm / (r * r), 0.0));
    return transmittanceToTop(lut, r, l.y) * smoothstep(horizon - 0.004, horizon + 0.004, l.y);
}

// The multiple-scattering LUT: x the cosine of the sun zenith angle, y the height
vec2 multiScatteringUv(float r, float muS)
{
    return vec2(muS * 0.5 + 0.5, (r - kGroundRadiusKm) / (kTopRadiusKm - kGroundRadiusKm));
}

// The sky-view LUT's horizon: the elevation (rad, below 0) of the planet's limb seen from the radius r
float skyViewHorizon(float r)
{
    return -acos(clamp(kGroundRadiusKm / r, 0.0, 1.0));
}

// The sky-view LUT (computed at a radius whose horizon elevation is horizon, skyViewHorizon): x the azimuth
// (atan(z, x)), y the elevation with the horizon at 0.5 and more texels near it (Hillaire's non-linear mapping). The
// scene shaders take the LUT's horizon from the block (lighting.atmosphere.w): the LUT is rewritten only when its
// inputs changed (Saiga::Vulkan::Atmosphere, which has the CPU copy of this mapping)
vec2 skyViewUv(vec3 dir, float horizon)
{
    float el = asin(clamp(dir.y, -1.0, 1.0));
    float v;
    if (el >= horizon)
        v = 0.5 + 0.5 * sqrt(clamp((el - horizon) / (0.5 * kAtmPi - horizon), 0.0, 1.0));
    else
        v = 0.5 - 0.5 * sqrt(clamp((horizon - el) / (horizon + 0.5 * kAtmPi), 0.0, 1.0));
    // atan(0, 0) is undefined (GLSL): a vertical ray takes the azimuth 0 (CodeRabbit #83)
    float u = dir.x != 0.0 || dir.z != 0.0 ? atan(dir.z, dir.x) / (2.0 * kAtmPi) + 0.5 : 0.5;
    return vec2(u, v);
}

vec3 skyViewDir(vec2 uv, float horizon)
{
    float el;
    if (uv.y >= 0.5)
    {
        float t = (uv.y - 0.5) * 2.0;
        el      = horizon + t * t * (0.5 * kAtmPi - horizon);
    }
    else
    {
        float t = (0.5 - uv.y) * 2.0;
        el      = horizon - t * t * (horizon + 0.5 * kAtmPi);
    }
    float az = (uv.x - 0.5) * 2.0 * kAtmPi;
    return vec3(cos(el) * cos(az), sin(el), cos(el) * sin(az));
}

// The rows of the sky-view LUT (Atmosphere::kSkyViewHeight)
const float kSkyViewRows = 108.0;

// The LUT's v at the elevation 0 (the fog colour, lighting.glsl): at least the centre of the first sky row, so the
// bilinear lookup never mixes in the ground (a camera at the sea level has its horizon at 0)
float skyViewFogV(float horizon)
{
    return max(0.5 + 0.5 * sqrt(-horizon / (0.5 * kAtmPi - horizon)), 0.5 + 0.5 / kSkyViewRows);
}

// The sky's radiance (cd/m^2, without the sun and moon discs) in the direction dir (the LUT's horizon elevation)
vec3 skyRadiance(sampler2D skyView, vec3 dir, float horizon)
{
    return textureLod(skyView, skyViewUv(dir, horizon), 0.0).rgb / kSkyLutScale;
}

// ---- The 2D cloud layer (S36a1c, A4; Saiga::Vulkan::CloudLayer): a tiling coverage texture of kCloudTile metres, a
// layer at the altitude (m above the sea level) with the earth's curvature, its alpha fading over the outer 30% of the
// radius. Shared by the layer's pass (cloud_layer.frag), the environment capture, the cloud shadow (shadows.glsl) and
// the previews.
const float kCloudTile          = 16384.0;
const float kEarthRadiusM       = 6371000.0;
const float kCloudShadowOpacity = 0.85;

// The layer's opacity at the coverage texture's uv for the cloudiness (the noise's top (1 - cloudiness))
float cloudCoverageFromNoise(float n, float cloudiness)
{
    return cloudiness <= 0.0 ? 0.0 : smoothstep(1.0 - cloudiness - 0.1, 1.0 - cloudiness + 0.1, n);
}

// The alpha of the layer's edge at the fraction of its radius (no visible rim)
float cloudEdgeAlpha(float r01)
{
    return 1.0 - smoothstep(0.7, 1.0, r01);
}

// The radiance of the layer (cd/m^2) seen along v: sunE the illuminance of the shadow light at the layer (after the
// transmittance) from the direction l, skyE and groundE the irradiance from above and below, coverage the opacity,
// below: the viewer is below the layer. A slab: the side towards the light reflects it diffusely (low light grazes
// it), the other side shows what a dense cloud lets through; thin parts scatter forward around the light (the silver
// lining)
vec3 cloudLayerRadiance(vec3 sunE, vec3 l, vec3 v, vec3 skyE, vec3 groundE, float coverage, bool below)
{
    const float albedo = 0.9;
    bool lightAbove    = l.y > 0.0;
    float through      = below == lightAbove ? mix(0.75, 0.22, coverage) : 1.0;
    float grazing      = 0.35 + 0.65 * abs(l.y);
    vec3 direct        = sunE * (albedo / kAtmPi) * grazing * through;
    float c            = dot(v, l);
    direct += sunE * miePhase(c) * 0.12 * (1.0 - coverage);
    vec3 ambient = (below ? mix(skyE, groundE, 0.35) : skyE) * (albedo / kAtmPi);
    return direct + ambient;
}

// The cloud layer's shadow (the factor of the shadow light's direct light) at a receiver: relXZ its camera-relative
// horizontal position, heightM its height above the sea level, l towards the light; origin the coverage texture's
// position of the camera (m, wrapped), clouds (altitude, radius, cloudiness, shadow fade), mean the mean coverage at
// the cloudiness. The receiver is projected along l to the layer; at or above the layer (with a 50 m blend below) the
// factor is 1; beyond the fade distance of the projected point the mean coverage replaces the texture
float cloudShadowAt(sampler2D coverage, vec2 origin, vec2 relXZ, float heightM, vec3 l, vec4 clouds, float mean)
{
    if (clouds.z <= 0.0 || l.y <= 0.0) return 1.0;
    float above = smoothstep(clouds.x - 50.0, clouds.x, heightM);
    vec2 p      = relXZ + (clouds.x - heightM) * l.xz / max(l.y, 0.02);
    float fade  = smoothstep(0.7 * clouds.w, clouds.w, length(p));
    float c     = cloudCoverageFromNoise(textureLod(coverage, (origin + p) / kCloudTile, 2.0).r, clouds.z);
    float occl  = mix(c, mean, fade) * kCloudShadowOpacity;
    return mix(1.0 - occl, 1.0, above);
}

#endif
