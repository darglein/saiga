/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430


// layout(binding=0, rgba16f) uniform image2D inputTex;
layout(binding=1, rgba16f) uniform image2D destTex;

layout(location = 5) uniform sampler2D inputTex;

layout(location = 7) uniform int layer;
layout(location = 8) uniform int num_layers;

layout(local_size_x = 16, local_size_y = 16) in;

#include "compute_helper.glsl"


vec4 DownsampleBox (vec2 uv) {
    return texture(inputTex, uv);
    vec2 texel_size = 1.f / textureSize(inputTex, 0);

    float d = 0.25;
    vec4 o = texel_size.xyxy * vec2(-d, d).xxyy;
    vec4 s =
    texture(inputTex, uv + o.xy) + texture(inputTex, uv + o.zy) +
    texture(inputTex, uv + o.xw) + texture(inputTex, uv + o.zw);
    return s * 0.25f;
}

void main() {
    ivec2 texel_position_small = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size_small = imageSize(destTex);
    if (texel_position_small.x >= image_size_small.x || texel_position_small.y >= image_size_small.y)
    {
        return;
    }

//    vec2 texel_position_large = (vec2(texel_position_small) * 0.5) + vec2(0.25);
//    ivec2 image_size_large = textureSize(inputTex, 0);
//    vec2 uv = Texel2UV(texel_position_large, image_size_large);
//    vec3 hdr_value = texture(inputTex, uv).rgb;

    vec2 uv;
    vec3 hdr_value;
    uv = Texel2UV(texel_position_small, image_size_small);
    hdr_value = DownsampleBox(uv).rgb;

    if(layer == num_layers-1) hdr_value *= (1.f / num_layers);

    hdr_value += imageLoad(destTex, texel_position_small).rgb * (1.f / num_layers) ;//* exp(layer/ float(num_layers) );
    //hdr_value *= 0.5;

    //hdr_value = imageLoad(destTex, texel_position_small).rgb ;

    imageStore(destTex, texel_position_small, vec4(hdr_value, 1));
}
