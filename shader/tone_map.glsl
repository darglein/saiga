/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430


#ifndef INPUT_TYPE
#define INPUT_TYPE rgba16f
#endif

#include "include/saiga/tone_map_operators.h"

layout(binding=0, INPUT_TYPE) uniform image2D inputTex;
layout(binding=1, rgba8) uniform image2D destTex;
// layout(binding=2, r32f) uniform image1D camera_response_tex;
layout(location = 2) uniform sampler1D camera_response_tex;

layout(local_size_x = 16, local_size_y = 16) in;

layout(location = 3) uniform float gamma = 1;
layout(location = 4) uniform int operator = 0;
// ====================================================================================



vec2 NormalizedUV(ivec2 texel, ivec2 size)
{
    float max_size = max(size[0], size[1]);
    vec2 center_pixel = vec2(size) / 2.f;
    vec2 centered_uv = (vec2(texel) -center_pixel) / max_size * 2;
    return centered_uv;
}


vec3 TonemapTexture(vec3 color, sampler1D tonemap_tex){
    // Camera response / Gamma
    color.x = texture(tonemap_tex, color.x).r;
    color.y = texture(tonemap_tex, color.y).r;
    color.z = texture(tonemap_tex, color.z).r;
    return color;
}

void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(inputTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        return;
    }

    vec3 hdr_value = imageLoad(inputTex, texel_position).rgb;

    vec3 ldr_value;


    if (operator == 0)
        ldr_value = TonemapGamma(hdr_value, gamma);
    else if (operator == 1)
        ldr_value = TonemapTexture(hdr_value, camera_response_tex);
    else if (operator == 2)
        ldr_value = TonemapReinhard(hdr_value, gamma);
    else if (operator == 3)
        ldr_value = TonemapUE3(hdr_value);
    else if (operator == 4)
        ldr_value = TonemapGamma(Tonemap_Filmic_UC2Default(hdr_value), gamma);
    else if (operator == 5)
        ldr_value = TonemapDrago(hdr_value, gamma);
    else
        ldr_value = vec3(0);


    //
    // ldr_value = TonemapPhotographic(hdr_value, gamma);
    //ldr_value = Tonemap_Filmic_UC2DefaultToGamma(hdr_value);


    imageStore(destTex, texel_position, vec4(ldr_value, 1));
}
