/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_VERTEX_SHADER

#version 330
#extension GL_ARB_explicit_uniform_location : enable

layout(location=0) in vec3 in_position;

#include "camera.glsl"

out vec2 tc;


void main() {
    tc = vec2(in_position.x,in_position.y);
    tc = tc * 0.5f + 0.5f;
    gl_Position = vec4(in_position.x,in_position.y,0,1);
}

##GL_FRAGMENT_SHADER

#version 330
#extension GL_ARB_explicit_uniform_location : enable

in vec2 tc;
uniform sampler2D image;

layout(location=0) out vec4 out_color;

layout(location=0) uniform int flip_y = 0;

void main() {
    vec2 tc2 = tc;
    if(flip_y == 1)
    {
        tc2.y = 1.0 - tc2.y;
    }
    out_color = texture( image, tc2 );
}


