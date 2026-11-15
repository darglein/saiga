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

// Exponential height fog: the density decreases with the height above the water level.
vec3 applyFog(vec3 color, vec3 pos)
{
    if (world.fog.z < 0.5) return color;
    vec3 ro    = world.cameraPos.xyz;
    vec3 rd    = pos - ro;
    float dist = length(rd);
    rd /= max(dist, 1e-4);
    float a  = world.fog.x;
    float b  = world.fog.y;
    float h0 = max(ro.y - world.fog.w, 0.0);
    // Integral of a * exp(-b * height) along the ray
    float k        = rd.y * b * dist;
    float integral = abs(k) > 1e-4 ? (1.0 - exp(-k)) / k : 1.0;
    float optical  = a * exp(-h0 * b) * dist * integral;
    float f        = 1.0 - exp(-optical);
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
