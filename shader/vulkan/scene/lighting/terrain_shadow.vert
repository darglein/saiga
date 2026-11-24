#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// The terrain casters of the cascades (S36a2, A5; TerrainPass::renderShadow): terrain.vert's vertex function with the
// main camera's LOD and morph (the frame's TerrainFrame: the shadow mesh equals the receiver mesh) and the cascade's
// camera-relative light view-projection, pancaked onto depth 0 (shadow_depth.vert). The node instances are the
// cascade's selection (TerrainPass::selectForShadow) in the shadow node buffer (binding 5 of the shadow set).

#include "../terrain_common.glsl"
#include "../terrain_heights.glsl"
#include "../terrain_vertex.glsl"

layout(push_constant) uniform PushConstants
{
    mat4 viewProj;
    int gridN;
    int pad0, pad1, pad2;
#ifdef ATLAS_DEPTH
    // S36b: xyz the light (camera-relative), w 1 / range (atlas_depth.frag)
    vec4 light;
#endif
}
pc;

#ifdef ATLAS_DEPTH
// S36b: the shadow atlas's faces (the nodes of the atlas selection, a fixed level without morph): the camera-relative
// position for atlas_depth.frag's linear depth; a perspective face: no pancaking
layout(location = 3) out vec3 outRelPos;
#endif

void main()
{
    int ext         = pc.gridN + 3;
    ivec2 e         = ivec2(gl_VertexIndex % ext, gl_VertexIndex / ext);
    ivec2 g         = clamp(e - 1, ivec2(0), ivec2(pc.gridN));
    bool skirt      = e.x == 0 || e.y == 0 || e.x == pc.gridN + 2 || e.y == pc.gridN + 2;
    TerrainVertex v = terrainVertex(uint(gl_InstanceIndex), g, skirt);
    gl_Position     = pc.viewProj * vec4(v.position, 1.0);
#ifdef ATLAS_DEPTH
    outRelPos = v.position;
#else
    gl_Position.z = max(gl_Position.z, 0.0);
#endif
}
