#version 450
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The sky of S20 (a copy of world/sky.vert): a full screen triangle at the far plane (depth 0 with reversed depth)
layout(location = 0) out vec2 outNdc;

void main()
{
    vec2 p      = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    outNdc      = p * 2.0 - 1.0;
    gl_Position = vec4(outNdc, 0.0, 1.0);
}
