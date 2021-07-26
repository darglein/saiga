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
    gl_Position =  proj * model * vec4(in_position.x,in_position.y,0,1);
//    gl_Position = vec4(in_position.x,in_position.y,0,1);
}





##GL_FRAGMENT_SHADER

#version 330
uniform mat4 model;

uniform sampler2D text;


in vec2 texCoord;

out vec4 out_color;


float linearDepth(float d){

    float f=60.0f;
    float n = 1.0f;
    return(2 * n) / (f + n - d * (f - n));
}


void main() {
    float d = texture(text,texCoord).r;
    out_color =  vec4(vec3(linearDepth(d)),1);
}


