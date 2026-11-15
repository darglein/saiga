/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Shading of the world: a uniform gray ("clay") with sun + hemisphere light, height fog.
// Requires common.glsl.

vec3 clayLighting(vec3 N, float ao)
{
    vec3 L        = normalize(world.sunDir.xyz);
    float diffuse = max(dot(N, L), 0.0) * world.sunDir.w;
    // Hemisphere ambient: brighter from above
    float hemi = world.shading.y * (0.6 + 0.4 * N.y);
    return vec3(world.shading.x * (diffuse + hemi) * ao);
}

// Optical depth of a ray segment of length L above the water level whose heights above the water go from hA to hB
// (both >= 0): the integral of a * exp(-b * height), computed from the bounded densities at both ends (no exp of a
// large positive number, which overflowed for a high camera looking down).
float fogAbove(float L, float hA, float hB, float a, float b)
{
    float eA = exp(-b * hA);
    float eB = exp(-b * hB);
    float dh = hB - hA;
    return abs(b * dh) > 1e-4 ? a * L * (eA - eB) / (b * dh) : a * L * 0.5 * (eA + eB);
}

// Exponential height fog: the density a * exp(-b * height) decreases with the height above the water level and is
// constant (a) below it. relPos: the position relative to the camera.
vec3 applyFog(vec3 color, vec3 relPos)
{
    if (world.fog.z < 0.5) return color;
    float dist = length(relPos);
    float a    = world.fog.x;
    float b    = world.fog.y;
    // Signed heights above the water level at both ends of the ray
    float y0 = world.cameraPos.y - world.fog.w;
    float y1 = y0 + relPos.y;
    float optical;
    if (y0 <= 0.0 && y1 <= 0.0)
    {
        optical = a * dist;
    }
    else if (y0 > 0.0 && y1 > 0.0)
    {
        optical = fogAbove(dist, y0, y1, a, b);
    }
    else
    {
        // The ray crosses the water level: constant density below, exponential above
        float below = dist * (y0 <= 0.0 ? -y0 : y1) / abs(y1 - y0);
        optical     = a * below + fogAbove(dist - below, 0.0, max(y0, y1), a, b);
    }
    float f = 1.0 - exp(-optical);
    return mix(color, world.fogColor.rgb, clamp(f, 0.0, 1.0));
}

vec3 lodTint(float lod)
{
    const vec3 colors[6] = vec3[](vec3(1.0, 0.45, 0.45), vec3(0.45, 1.0, 0.45), vec3(0.5, 0.55, 1.0),
                                  vec3(1.0, 1.0, 0.45), vec3(1.0, 0.45, 1.0), vec3(0.45, 1.0, 1.0));
    return colors[int(max(lod, 0.0)) % 6];
}

vec4 finalColor(vec3 c, float alpha)
{
    c = clamp(c, 0.0, 1.0);
    // The swap chain is UNORM: encode here
    if (world.shading.w > 0.5) c = pow(c, vec3(1.0 / 2.2));
    return vec4(c, alpha);
}
