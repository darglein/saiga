#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

#include "terrain_common.glsl"
#include "terrain_heights.glsl"
#include "terrain_vertex.glsl"

// The grid of a node: (gridN + 3)^2 vertices, from gl_VertexIndex; the outer ring is the skirt (hangs below the edge
// vertex). The instance is the node (gl_InstanceIndex includes the draw's first instance)
layout(push_constant) uniform PushConstants
{
    int gridN;
    int pad;
}
pc;

// Position relative to the camera
layout(location = 0) out vec3 outPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out float outLod;
// 1 at the skirt ring, 0 at the grid: > 0 in every fragment of a skirt triangle (the shading of the skirt)
layout(location = 3) out float outSkirt;

// The depth pre-pass draws the terrain with this shader alone: the scene pass's EQUAL test needs bit-identical depths
invariant gl_Position;

void main()
{
    int ext      = pc.gridN + 3;
    ivec2 e      = ivec2(gl_VertexIndex % ext, gl_VertexIndex / ext);
    ivec2 g      = clamp(e - 1, ivec2(0), ivec2(pc.gridN));
    bool skirt   = e.x == 0 || e.y == 0 || e.x == pc.gridN + 2 || e.y == pc.gridN + 2;
    TerrainVertex v = terrainVertex(uint(gl_InstanceIndex), g, skirt);
    outPos       = v.position;
    outNormal    = v.normal;
    outLod       = v.lod;
    outSkirt     = skirt ? 1.0 : 0.0;
    gl_Position  = frame.viewProjRel * vec4(v.position, 1.0);
}
