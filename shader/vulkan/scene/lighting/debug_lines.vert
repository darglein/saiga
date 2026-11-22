#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// DebugLines (S36, A8): camera-relative line lists (the positions made relative to the camera in double on the CPU)
// with a display colour (sRGB) per vertex.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(push_constant) uniform Push
{
    mat4 viewProj;
    // > 0: the scene writes linear values: srgbToLinear(colour) * scale; 0: display values
    float scale;
}
pc;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor    = inColor;
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
}
