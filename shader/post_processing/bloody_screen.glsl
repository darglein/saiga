/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "post_processing_vertex_shader.glsl"


##GL_FRAGMENT_SHADER
#version 330

uniform float intensity;

#include "post_processing_helper_fs.glsl"


float intensityRect(vec2 tc){
    vec2 d = abs(tc-vec2(0.5f));
    return clamp(max(d.x,d.y),0,1);
}

vec4 bloodyScreenColor(vec2 tc, vec3 backgroundColor){
    float a = intensityRect(tc);
    a = smoothstep(0.3,0.5,a);
    return vec4(backgroundColor + intensity*vec3(a,0,0),1);
}

void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec3 c = texelFetch( image, tci ,0).rgb;
//    vec3 c = texture( image, tc ).rgb;	
    out_color = bloodyScreenColor(tc,c);
}


