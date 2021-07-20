/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430


//layout(binding=0, rgba16f) uniform image2D inputTex;
layout(location = 5) uniform sampler2D inputTex;
layout(binding=1, rgba16f) uniform image2D destTex;

layout(location = 0) uniform float exposure = 1;

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
#include "compute_helper.glsl"

// ====================================================================================

vec4 DownsampleBox (vec2 uv) {
    vec2 texel_size = 1.f / textureSize(inputTex, 0);

    float d = 0;
    vec4 o = texel_size.xyxy * vec2(-d, d).xxyy;
    vec4 s =
    texture(inputTex, uv + o.xy) + texture(inputTex, uv + o.zy) +
    texture(inputTex, uv + o.xw) + texture(inputTex, uv + o.zw);
    return s * 0.25f;
}

void main() {
    ivec2 texel_position = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(destTex);
    if (texel_position.x >= image_size.x || texel_position.y >= image_size.y)
    {
        return;
    }

    vec2 uv = Texel2UV(texel_position, image_size);
    vec3 hdr_value = texture(inputTex, uv).rgb;

    if (false)
    {
        hdr_value = max(hdr_value - vec3(params.bloom_threshold), vec3(0));
    } else
    {
        float luminance = dot(hdr_value, vec3(1)) * 0.3333;
        if (luminance > params.bloom_threshold)
        {
            // preserve luminance by rescaling the hdr value to the cutoff luminance
            float remaining_luminance = luminance - params.bloom_threshold;
            hdr_value = hdr_value * (remaining_luminance / luminance);
        } else {
            hdr_value = vec3(0);
        }

    }

    imageStore(destTex, texel_position, vec4(hdr_value, 1));
}
