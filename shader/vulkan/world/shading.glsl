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

// Exponential height fog: the density decreases with the height above the water level. relPos: the position
// relative to the camera.
vec3 applyFog(vec3 color, vec3 relPos)
{
    if (world.fog.z < 0.5) return color;
    vec3 ro    = world.cameraPos.xyz;
    vec3 rd    = relPos;
    float dist = length(rd);
    rd /= max(dist, 1e-4);
    float a = world.fog.x;
    float b = world.fog.y;
    // Heights above the water level at both ends of the ray (the density is constant below the water level)
    float h0 = max(ro.y - world.fog.w, 0.0);
    float h1 = max(ro.y + rd.y * dist - world.fog.w, 0.0);
    // Integral of a * exp(-b * height) along the ray, from the bounded densities at both ends (no exp of a large
    // positive number, which overflowed for a high camera looking down)
    float e0      = exp(-b * h0);
    float e1      = exp(-b * h1);
    float dh      = h1 - h0;
    float optical = abs(b * dh) > 1e-4 ? a * dist * (e0 - e1) / (b * dh) : a * dist * 0.5 * (e0 + e1);
    float f       = 1.0 - exp(-optical);
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
