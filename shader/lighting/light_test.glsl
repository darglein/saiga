/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;


#include "camera.glsl"
uniform mat4 model;


void main() {
    gl_Position = vec4(in_position.x,in_position.y,0,1);

}

##GL_FRAGMENT_SHADER
#version 330


uniform sampler2D image;



out vec4 out_color;

void main() {
    ivec2 tci = ivec2(gl_FragCoord.xy);
    out_color = texelFetch( image, tci ,0);
}


