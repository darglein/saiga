/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The stars and the moon disc of the sky pass (S36, A4). Stars: a hashed grid on the faces of a cube in the celestial
// frame (rotated by the hour angle, starRotation), at most one star per cell with a random position and brightness, a
// disc at least 1.5 px wide (its radiance scaled so its illuminance does not depend on the size; no twinkle). The moon:
// a disc of the moon's angular radius, a sphere lit by the sun direction (the phase follows from the geometry), with a
// radiance that gives the full moon's illuminance.
#ifndef STARS_GLSL
#define STARS_GLSL

uint starHash(uvec3 v)
{
    uint h = v.x * 0x8da6b343u ^ v.y * 0xd8163841u ^ v.z * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float starRandom(uint h)
{
    return float(h & 0xFFFFFFu) / 16777216.0;
}

// The stars of one cube face's cells around the celestial direction d; uv: d projected onto the face (its axis
// component scaled to 1). A star is its face and cell (its hash), so evaluating several faces counts none twice
vec3 starFaceRadiance(vec3 d, uint face, vec2 uv, float pixel)
{
    const float cells = 160.0;
    vec2 g            = (uv * 0.5 + 0.5) * cells;
    // Only within a cell of the face: the neighbour search reaches no further
    if (any(lessThan(g, vec2(-1.0))) || any(greaterThan(g, vec2(cells + 1.0)))) return vec3(0.0);
    vec3 result = vec3(0.0);
    // The cell and its neighbours (a star near a cell border)
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
        {
            vec2 cell = floor(g) + vec2(i, j);
            if (cell.x < 0.0 || cell.y < 0.0 || cell.x >= cells || cell.y >= cells) continue;
            uint h = starHash(uvec3(uvec2(cell), face));
            if (starRandom(h) > 0.07) continue;
            vec2 pos  = cell + vec2(starRandom(h * 747796405u + 1u), starRandom(h * 2891336453u + 7u));
            vec2 puv  = pos / cells * 2.0 - 1.0;
            vec3 sdir = face == 0u   ? vec3(1.0, puv.y, puv.x)
                        : face == 1u ? vec3(-1.0, puv.y, puv.x)
                        : face == 2u ? vec3(puv.x, 1.0, puv.y)
                        : face == 3u ? vec3(puv.x, -1.0, puv.y)
                        : face == 4u ? vec3(puv.x, puv.y, 1.0)
                                     : vec3(puv.x, puv.y, -1.0);
            sdir      = normalize(sdir);
            // The illuminance (lux): a steep magnitude distribution, the brightest about 1e-5 lux
            float m = starRandom(h * 1103515245u + 12345u);
            float E = 8e-6 * pow(m, 8.0) + 2e-8;
            // Colour temperature: blue-white to orange
            float k      = starRandom(h * 22695477u + 3u);
            vec3 tint    = mix(vec3(1.0, 0.82, 0.6), vec3(0.75, 0.85, 1.0), k);
            float radius = 1.5 * pixel;
            float dist   = acos(clamp(dot(d, sdir), -1.0, 1.0));
            float w      = 1.0 - smoothstep(0.5 * radius, radius, dist);
            result += tint * (E * w / (kAtmPi * radius * radius * 0.6));
        }
    return result;
}

// The stars' radiance (cd/m^2) in the celestial direction d; pixel: the angular size of a pixel (rad); intensity: the
// brightness multiplier (EnvironmentState::starIntensity)
vec3 starRadiance(vec3 d, float pixel, float intensity)
{
    if (intensity <= 0.0) return vec3(0.0);
    // The face of d and the faces whose edge is near (CodeRabbit #83: a star disc next to a cube edge was cut off on
    // the other face's side; the 3 x 3 cells reach one cell, about 0.7 degrees, and 0.9 of the major component keeps
    // every face within it)
    vec3 a      = abs(d);
    float major = max(a.x, max(a.y, a.z));
    vec3 result = vec3(0.0);
    if (a.x >= 0.9 * major) result += starFaceRadiance(d, d.x > 0.0 ? 0u : 1u, d.zy / a.x, pixel);
    if (a.y >= 0.9 * major) result += starFaceRadiance(d, d.y > 0.0 ? 2u : 3u, d.xz / a.y, pixel);
    if (a.z >= 0.9 * major) result += starFaceRadiance(d, d.z > 0.0 ? 4u : 5u, d.xy / a.z, pixel);
    return result * intensity;
}

// The moon disc's radiance (cd/m^2) in the view direction d: moonDir towards the moon, sunDir towards the sun; the
// radiance of the fully lit disc gives 0.27 lux over its solid angle
vec3 moonDiscRadiance(vec3 d, vec3 moonDir, vec3 sunDir, float pixel)
{
    const float radius = 0.0045;  // rad
    float c            = dot(d, moonDir);
    float edge         = cos(radius);
    float soft         = max(pixel, 1e-5);
    float disc         = smoothstep(edge - soft * radius, edge + soft * radius * 0.2, c);
    if (disc <= 0.0) return vec3(0.0);
    // The sphere's normal at the disc point (towards the viewer side)
    vec3 tangent = d - moonDir * c;
    float s      = clamp(length(tangent) / sin(radius), 0.0, 1.0);
    vec3 n       = -moonDir * sqrt(max(1.0 - s * s, 0.0)) + (s > 0.0 ? normalize(tangent) * s : vec3(0.0));
    float lit    = max(dot(n, sunDir), 0.0);
    // Full moon: lit = sqrt(1 - s^2) on average 2/3 over the disc
    float omega = kAtmPi * radius * radius;
    return vec3(0.98, 0.96, 0.92) * (0.27 / omega * 1.5 * lit * disc);
}

#endif
