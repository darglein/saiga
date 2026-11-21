/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The GPU terrain set (S20, Saiga::Vulkan::TerrainAtlas): int32 heights in height units (kHeightUnit) in the atlas
// (one layer of samplesPerSide^2 per tile), the 32 m pyramid, the page table (per streamed level a grid of layers, -1:
// none), the grass masks (bytes, 4 per uint) and the frame's node instances. Requires terrain_common.glsl.

// A node instance (32 bytes, TerrainPass::GpuNode): the origin relative to the anchor (integer metres), the global
// sample index of the origin at the node's level, level | quarter << 8 | water << 9, the skirt depth (m). Water: sampleX
// is the grid spacing (m)
struct TerrainNode
{
    int originX, originZ;
    int sampleX, sampleZ;
    uint levelFlags;
    float skirtDepth;
    uint pad0, pad1;
};

layout(std430, set = 0, binding = 1) readonly buffer TerrainAtlasBuffer
{
    int atlasHeights[];
};
layout(std430, set = 0, binding = 2) readonly buffer TerrainPyramidBuffer
{
    int pyramidHeights[];
};
layout(std430, set = 0, binding = 3) readonly buffer TerrainPageBuffer
{
    int pageTable[];
};
layout(std430, set = 0, binding = 4) readonly buffer TerrainMaskBuffer
{
    uint maskWords[];
};
layout(std430, set = 0, binding = 5) readonly buffer TerrainNodeBuffer
{
    TerrainNode nodes[];
};

int pyramidSample(ivec2 p)
{
    int P = frame.grid.z;
    p     = clamp(p, ivec2(0), ivec2(P - 1));
    return pyramidHeights[p.y * P + p.x];
}

// The atlas layer of a tile of a streamed level, -1 if none
int pageLayer(int level, ivec2 tile)
{
    ivec4 L = frame.levels[level];
    if (any(lessThan(tile, ivec2(0))) || any(greaterThanEqual(tile, ivec2(L.y)))) return -1;
    return pageTable[L.x + tile.y * L.y + tile.x];
}

int layerSample(int layer, ivec2 local)
{
    int n = frame.grid.x;
    return atlasHeights[layer * n * n + local.y * n + local.x];
}

// A sample of a level (global sample index s at that level) wherever it is resident: its tile, else the next coarser
// resident level (rounded down: a possible shading seam, never a crack), else the pyramid
int anyHeight(int level, ivec2 s)
{
    int S  = frame.grid.y;
    int n1 = frame.grid.x - 1;
    if (level >= S) return pyramidSample(s << (level - S));
    for (int k = level; k < S; ++k)
    {
        ivec2 tiles = ivec2(frame.levels[k].y);
        ivec2 sc    = clamp(s, ivec2(0), tiles * n1);
        ivec2 tile  = min(sc / n1, tiles - 1);
        int layer   = pageLayer(k, tile);
        if (layer >= 0) return layerSample(layer, sc - tile * n1);
        s = s >> 1;
    }
    return pyramidSample(s);
}

// The height of the grid sample g (0..gridN) of a node at its level: its own tile (GPU-resident: the node was
// selected) or the pyramid
int nodeHeight(TerrainNode nd, int level, ivec2 g)
{
    int S   = frame.grid.y;
    ivec2 s = ivec2(nd.sampleX, nd.sampleZ) + g;
    if (level >= S) return pyramidSample(s << (level - S));
    int n1     = frame.grid.x - 1;
    ivec2 tile = ivec2(nd.sampleX, nd.sampleZ) / n1;
    return layerSample(pageLayer(level, tile), s - tile * n1);
}

// The grass mask texel (0..1) of a cell of a mask layer
float maskDensity(int layer, ivec2 cell)
{
    int m       = 256 / frame.grid.w;
    int index   = layer * m * m + cell.y * m + cell.x;
    uint word   = maskWords[index >> 2];
    return float((word >> (8 * (index & 3))) & 0xFFu) / 255.0;
}
