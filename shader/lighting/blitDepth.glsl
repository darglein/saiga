/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec2 in_tex;


#include "camera.glsl"
uniform mat4 model;


out vec2 texCoord;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    gl_Position = vec4(in_position.x,in_position.y,1,1);
}





##GL_FRAGMENT_SHADER

#version 330

//uniform sampler2DMS image;
uniform sampler2D image;

in vec2 texCoord;

layout(location=0) out vec4 out_color;

void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    float d = texelFetch( image, tci ,0).r;

//     float d = texture(image, texCoord).r;
    gl_FragDepth = d;
    out_color = vec4(0);
}


