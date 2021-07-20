/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "post_processing_helper_fs.glsl"

vec4 fxaaLumaRGB(vec3 color){
    return vec4(color,dot(color, vec3(0.299, 0.587, 0.114)));
}

vec4 fxaaLumaLinearRGB(vec3 color){
    //fxaa needs luma in non linear color space
    //because srgb is linear we need sqrt here
    return vec4(color,sqrt(dot(color, vec3(0.299, 0.587, 0.114))));
}


void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec4 color = texelFetch( image, tci ,0);
//    vec4 color = texture( image, tc );
    out_color = fxaaLumaLinearRGB(color.rgb);

}


