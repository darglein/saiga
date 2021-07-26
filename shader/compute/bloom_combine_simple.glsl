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

layout(binding=0, INPUT_TYPE) uniform image2D inputTex;
// layout(binding=1, rgba16) uniform image2D destTex;

layout(location = 0) uniform sampler2D blur1;
layout(location = 1) uniform sampler2D blur2;
layout(location = 2) uniform sampler2D blur3;
layout(location = 3) uniform sampler2D blur4;
layout(location = 4) uniform sampler2D blur5;

layout(local_size_x = 16, local_size_y = 16) in;

struct BloomParameters
{
    float bloom_threshold;
    float bloom_strength;
    int levels;
    int flags;
};

layout (std140, binding = 3) uniform lightDataBlockPoint
{
    BloomParameters params;
};

// ====================================================================================

#include "compute_helper.glsl"


void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(inputTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        return;
    }
    vec2 uv = Texel2UV(texel_position, image_size);

    vec3 hdr_value = imageLoad(inputTex, texel_position).rgb;


    vec3 bloom = texture(blur1, uv).rgb ;
//    vec3 bloom = texture(blur1, uv).rgb + texture(blur2, uv).rgb + texture(blur3, uv).rgb;
    vec3 combined = hdr_value + bloom * params.bloom_strength;

     // combined = texture(blur3, uv).rgb;
//    combined =  texture(blur1, uv).rgb + texture(blur2, uv).rgb + texture(blur3, uv).rgb;
     imageStore(inputTex, texel_position, vec4(combined, 1));
//    imageStore(inputTex, texel_position, vec4(hdr_value, 1));
}
