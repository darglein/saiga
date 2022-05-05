/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_color;

#include "camera.glsl"
uniform mat4 model;

out vec3 color;

void main() {
    color = in_color.rgb;
    gl_Position = viewProj *model* vec4(in_position.xyz,1);
}





##GL_FRAGMENT_SHADER

#version 330

in vec3 color;



layout(location=0) out vec4 out_color;

void main() {
    out_color = vec4(color,1);
}


