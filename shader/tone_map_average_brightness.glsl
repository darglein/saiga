/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER

#version 430
#extension GL_NV_shader_atomic_float : enable

struct TonemapParameters
{
    vec4 white_point_exposure;
    vec4 vignette_coeffs;
    vec2 vignette_offset;
    vec2 padding;

};

struct TonemapTempParameters
{
    // (r,g,b,l)
    vec4 average_color_luminace;
};

layout (std140, binding = 3) uniform lightDataBlockPoint
{
    TonemapParameters params;
};

layout (std140, binding = 4) buffer lightDataTmpBlockPoint
{
    TonemapTempParameters temp_params;
};


layout(binding=0, rgba16f) uniform image2D inputTex;
layout(local_size_x = 16, local_size_y = 16) in;

shared int pixel_count;
shared vec4 average_color_luminace;

void main() {
    int location = int(gl_LocalInvocationIndex);
    vec3 color = vec3(0);
    int count;

    int num_blocks = int(gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z);
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(inputTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        count = 0;
    }else{
        color = imageLoad(inputTex, texel_position).rgb;
        count = 1;
    }

    if(location == 0)
    {
        pixel_count = 0;
        average_color_luminace = vec4(0);
    }

    memoryBarrier();
    barrier();

     float brightness = max(color.x, max( color.y , color.z));
    // float brightness = dot(color, vec3(1)) / 3.f;
//    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));

    vec4 c_l = vec4(color , brightness);

    atomicAdd(pixel_count, count);

    atomicAdd(average_color_luminace[0], c_l[0]);
    atomicAdd(average_color_luminace[1], c_l[1]);
    atomicAdd(average_color_luminace[2], c_l[2]);
    atomicAdd(average_color_luminace[3], c_l[3]);

    memoryBarrier();
    barrier();

    if(location == 0)
    {
//        float block_brightness = brightness_sum / pixel_count;
        // atomicAdd(temp_params.mean_brightness, block_brightness / num_blocks);

        c_l = average_color_luminace / pixel_count / num_blocks;

        atomicAdd(temp_params.average_color_luminace[0], c_l[0]);
        atomicAdd(temp_params.average_color_luminace[1], c_l[1]);
        atomicAdd(temp_params.average_color_luminace[2], c_l[2]);
        atomicAdd(temp_params.average_color_luminace[3], c_l[3]);
    }

}
