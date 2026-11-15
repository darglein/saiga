/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The terrain height map (R32Sfloat, texel (i, j) at the world position (i, j) * spacing). Interpolated like the
// terrain mesh and Saiga::Vulkan::HeightField::height(). Requires common.glsl.
layout(set = 0, binding = 1) uniform sampler2D heightMap;

float heightTexel(ivec2 c)
{
    int n = int(world.terrain.x);
    return texelFetch(heightMap, clamp(c, ivec2(0), ivec2(n - 1)), 0).r;
}

float heightAt(vec2 p)
{
    float n = world.terrain.x - 1.0;
    vec2 g  = clamp(p / world.terrain.y, vec2(0.0), vec2(n));
    vec2 i  = min(floor(g), vec2(n - 1.0));
    vec2 f  = g - i;
    ivec2 c = ivec2(i);
    float ha = heightTexel(c);
    float hb = heightTexel(c + ivec2(1, 0));
    float hc = heightTexel(c + ivec2(0, 1));
    float hd = heightTexel(c + ivec2(1, 1));
    // The cell is split along the diagonal (1,0)-(0,1)
    if (f.x + f.y <= 1.0) return ha + f.x * (hb - ha) + f.y * (hc - ha);
    return hd + (1.0 - f.x) * (hc - hd) + (1.0 - f.y) * (hb - hd);
}

vec3 normalAt(vec2 p, float e)
{
    float hl = heightAt(p - vec2(e, 0.0));
    float hr = heightAt(p + vec2(e, 0.0));
    float hd = heightAt(p - vec2(0.0, e));
    float hu = heightAt(p + vec2(0.0, e));
    return normalize(vec3(hl - hr, 2.0 * e, hd - hu));
}
