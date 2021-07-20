/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430

layout(binding=0, rgba16f) uniform image2D inputTex;
layout(binding=1, rgba16) uniform image2D destTex;

layout(local_size_x = 16, local_size_y = 16) in;

// ====================================================================================

void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    imageStore(destTex, texel_position, imageLoad(inputTex, texel_position));
}
