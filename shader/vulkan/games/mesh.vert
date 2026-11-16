#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// Flat colored low-poly meshes of the tech demo games (games/common/MeshRenderer, transitional until the engine's
// RenderScene). Positions are relative to the camera: the CPU computes the model matrix relative to the camera in
// double, so the precision does not depend on the distance to the origin.

#include "mesh_common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 outRelPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;

void main()
{
    vec4 rel    = pc.modelRel * vec4(inPosition, 1.0);
    outRelPos   = rel.xyz;
    outNormal   = mat3(pc.normal0.xyz, pc.normal1.xyz, pc.normal2.xyz) * inNormal;
    outColor    = inColor * pc.tint.rgb;
    gl_Position = game.viewProjRel * rel;
}
