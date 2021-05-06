/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER

#version 430


layout(binding=0, rgba16f) uniform image2D inputTex;
layout(binding=1, rgba8) uniform image2D destTex;
// layout(binding=2, r32f) uniform image1D camera_response_tex;
layout(location = 2) uniform sampler1D camera_response_tex;

layout(local_size_x = 16, local_size_y = 16) in;

// ====================================================================================



vec2 NormalizedUV(ivec2 texel, ivec2 size)
{
    float max_size = max(size[0], size[1]);
    vec2 center_pixel = vec2(size) / 2.f;
    vec2 centered_uv = (vec2(texel) -center_pixel) / max_size * 2;
    return centered_uv;
}


vec3 Tonemap(vec3 color, ivec2 texel, vec2 normalized_uv){
    // Camera response / Gamma
    color.x = texture(camera_response_tex, color.x).r;
    color.y = texture(camera_response_tex, color.y).r;
    color.z = texture(camera_response_tex, color.z).r;
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
    vec3 ldr_value = Tonemap(hdr_value, texel_position, NormalizedUV(texel_position, image_size));
    imageStore(destTex, texel_position, vec4(ldr_value, 1));
}
