/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The terrain vertex of S20 (terrain.vert and terrain_probe.comp): exact integer positions relative to the frame
// anchor minus one per-frame cameraRel, the heights relative to refHeight (int32 differences below 2^24: exact in
// float). A vertex shared by two nodes computes the same integers and the same morph factor: bit-identical. Requires
// terrain_common.glsl and terrain_heights.glsl.

struct TerrainVertex
{
    vec3 position;  // relative to the camera
    vec3 normal;
    float lod;    // the level, -1 for water
    float morph;  // the morph factor k (0 for water)
};

// g: the grid sample (0..gridN); skirt: hang the vertex skirtDepth below its edge vertex
TerrainVertex terrainVertex(uint nodeIndex, ivec2 g, bool skirt)
{
    TerrainNode nd = nodes[nodeIndex];
    int level      = int(nd.levelFlags & 0xFFu);
    bool water     = (nd.levelFlags & 0x200u) != 0u;
    ivec2 origin   = ivec2(nd.originX, nd.originZ);
    vec3 cr        = frame.cameraRel.xyz;
    TerrainVertex v;
    if (water)
    {
        // The plane at the sea level; sampleX: the grid spacing (m)
        precise vec2 xz = vec2(origin + g * nd.sampleX) - cr.xz;
        precise float y = float(frame.anchor.w - frame.anchor.z) * kHeightUnit - cr.y;
        v.position      = vec3(xz.x, y, xz.y);
        v.normal        = vec3(0, 1, 0);
        v.lod           = -1.0;
        v.morph         = 0.0;
        return v;
    }
    int spacing = frame.grid.w << level;
    // The morph target: rounded down to even (the coarse grid with the same diagonal; node origins are even)
    ivec2 gt         = g & ~1;
    precise vec2 xz  = vec2(origin + g * spacing) - cr.xz;
    precise vec2 xzt = vec2(origin + gt * spacing) - cr.xz;
    int h            = nodeHeight(nd, level, g);
    int ht           = nodeHeight(nd, level, gt);
    precise float hr  = float(h - frame.anchor.z);
    precise float hrt = float(ht - frame.anchor.z);
    // The morph factor from the LOD metric of the unmorphed vertex (CDLOD): sqrt(dxz^2 + hc^2) with the frame's camera
    // height above the surface under it (lod.z), as TerrainPass's selection; relief independent, so a coarse node at a
    // level border never morphs its edge. The morph is complete at 99 % of the range: the fine side of a border (beyond
    // the range in double) is fully morphed also with the float rounding of the distance
    precise float k = 0.0;
    // S36b: the shadow atlas's nodes of a fixed level (flag 0x400) do not morph
    if (level < frame.flags.y && (nd.levelFlags & 0x400u) == 0u)
    {
        precise float dist = length(vec3(xz.x, frame.lod.z, xz.y));
        float range        = frame.lod.x * exp2(float(level));
        float prev         = level > 0 ? range * 0.5 : 0.0;
        float start        = prev + (range - prev) * frame.lod.y;
        k                  = clamp((dist - start) / (range * 0.99 - start), 0.0, 1.0);
    }
    precise vec2 p = mix(xz, xzt, k);
    precise float y = mix(hr, hrt, k) * kHeightUnit - cr.y;
    if (skirt) y -= nd.skirtDepth;
    v.position = vec3(p.x, y, p.y);

    // The normal: central differences of the level's samples (across a non-resident neighbour the next coarser level)
    ivec2 s   = ivec2(nd.sampleX, nd.sampleZ) + g;
    float hl  = float(anyHeight(level, s - ivec2(1, 0)) - h);
    float hrr = float(anyHeight(level, s + ivec2(1, 0)) - h);
    float hd  = float(anyHeight(level, s - ivec2(0, 1)) - h);
    float hu  = float(anyHeight(level, s + ivec2(0, 1)) - h);
    v.normal  = normalize(vec3((hl - hrr) * kHeightUnit, 2.0 * float(spacing), (hd - hu) * kHeightUnit));
    v.lod     = float(level);
    v.morph   = k;
    return v;
}
