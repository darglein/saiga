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

uniform mat4 model;

#include "camera.glsl"

out vec2 texCoord;

void main() {
    texCoord = in_tex;
//    gl_Position = proj * model * vec4(in_position,1);
//    gl_Position =  model * vec4(in_position.x,in_position.y,0,1);
    gl_Position =  proj * model * vec4(in_position.x,in_position.y,0,1);
//    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 330

uniform sampler2D text;


in vec2 texCoord;

out vec4 out_color;

void main() {
    vec4 diffColor = texture(text,texCoord);
//    vec3 d = texture(text,texCoord).rgb;
    out_color =  diffColor;
}


