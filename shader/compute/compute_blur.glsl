/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430


#ifdef BLUR_X
vec2 blur_direction = vec2(1, 0);
#elif BLUR_Y
vec2 blur_direction = vec2(0, 1);
#else
#error Define either BLUR_X or BLUR_Y
#endif

#ifndef BLUR_SIZE
#define BLUR_SIZE 2
#endif


// layout(binding=0, rgba16f) uniform image2D inputTex;
layout(binding=1, rgba16) uniform image2D destTex;

layout(location = 5) uniform sampler2D inputTex;

layout(local_size_x = 16, local_size_y = 16) in;

// ====================================================================================

#include "compute_helper.glsl"

#if BLUR_SIZE == 2
uniform float offset[2] = float[](0.0, 1.3333333333333333);
uniform float weight[2] = float[](0.29411764705882354, 0.35294117647058826);
#elif BLUR_SIZE == 3
uniform float offset[3] = float[](0.0, 1.3846153846, 3.2307692308);
uniform float weight[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);
#elif BLUR_SIZE == 4
uniform float offset[4] = float[](0.0, 1.411764705882353, 3.2941176470588234, 5.176470588235294);
uniform float weight[4] = float[](0.1964825501511404, 0.2969069646728344, 0.09447039785044732, 0.010381362401148057);
#else
#error Undefined Blur Size
#endif

void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(destTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        return;
    }
    vec2 uv = Texel2UV(texel_position, image_size);


    vec3 out_color = texture(inputTex, uv).rgb * weight[0];

    for (int i=1; i<BLUR_SIZE; i++) {

        float off = offset[i];
        float w = weight[i];
        vec2 off2 = blur_direction * off;

        out_color += texture2D(inputTex, uv + (off2 / image_size)).rgb * w;
        out_color += texture2D(inputTex, uv - (off2 / image_size)).rgb * w;
    }

    imageStore(destTex, texel_position, vec4(out_color, 1));
}
