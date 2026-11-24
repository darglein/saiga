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

// The sky-view LUT at the radius r: x the azimuth (atan(z, x)), y the elevation with the horizon at 0.5 and more
// texels near it (Hillaire's non-linear mapping)
vec2 skyViewUv(vec3 dir, float r)
{
    float horizon = -acos(clamp(sqrt(max(r * r - kGroundRadiusKm * kGroundRadiusKm, 0.0)) / r, 0.0, 1.0));
    float el      = asin(clamp(dir.y, -1.0, 1.0));
    float v;
    if (el >= horizon)
        v = 0.5 + 0.5 * sqrt(clamp((el - horizon) / (0.5 * kAtmPi - horizon), 0.0, 1.0));
    else
        v = 0.5 - 0.5 * sqrt(clamp((horizon - el) / (horizon + 0.5 * kAtmPi), 0.0, 1.0));
    // atan(0, 0) is undefined (GLSL): a vertical ray takes the azimuth 0 (CodeRabbit #83)
    float u = dir.x != 0.0 || dir.z != 0.0 ? atan(dir.z, dir.x) / (2.0 * kAtmPi) + 0.5 : 0.5;
    return vec2(u, v);
}

vec3 skyViewDir(vec2 uv, float r)
{
    float horizon = -acos(clamp(sqrt(max(r * r - kGroundRadiusKm * kGroundRadiusKm, 0.0)) / r, 0.0, 1.0));
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

// The sky's radiance (cd/m^2, without the sun and moon discs) in the direction dir seen from the radius r
vec3 skyRadiance(sampler2D skyView, vec3 dir, float r)
{
    return textureLod(skyView, skyViewUv(dir, r), 0.0).rgb / kSkyLutScale;
}

#endif
