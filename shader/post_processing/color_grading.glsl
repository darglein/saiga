/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

#include "post_processing_helper_fs.glsl"

uniform sampler3D colorGradingLookupTable;
const float lutSize = 16.0;
const vec3 scale = vec3((lutSize - 1.0) / lutSize);
const vec3 offset = vec3(1.0 / (2.0 * lutSize));


void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec3 rgbM = texelFetch( image, tci ,0).rgb;
	out_color = vec4(texture(colorGradingLookupTable, scale * rgbM.xyz + offset).rgb,1);
}