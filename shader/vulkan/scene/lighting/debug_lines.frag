#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#extension GL_GOOGLE_include_directive : require

// DebugLines (S36, A8): the line's display colour, linear and scaled when the scene writes linear values (S17's
// contract: the tone map then shows about the colour).

#include "../color.glsl"

layout(push_constant) uniform Push
{
    mat4 viewProj;
    float scale;
}
pc;

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = pc.scale > 0.0 ? vec4(srgbToLinear(inColor.rgb) * pc.scale, 1.0) : vec4(inColor.rgb, 1.0);
}
