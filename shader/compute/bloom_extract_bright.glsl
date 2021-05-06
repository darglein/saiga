/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER

#version 430

layout(binding=0, rgba16f) uniform image2D inputTex;
layout(binding=1, rgba16) uniform image2D destTex;

layout(location = 0) uniform float exposure = 1;

layout(local_size_x = 16, local_size_y = 16) in;

// ====================================================================================

void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(inputTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        return;
    }

    vec3 hdr_value = imageLoad(inputTex, texel_position).rgb;

    float ex = 1.f/exposure;
    ex = 1;
    hdr_value = max(hdr_value - vec3(ex, ex, ex), vec3(0));


    imageStore(destTex, texel_position, vec4(hdr_value, 1));
}
