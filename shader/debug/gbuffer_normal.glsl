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

uniform sampler2D text;


in vec2 texCoord;

out vec4 out_color;


vec3 unpackNormal3 (vec2 enc)
{
    vec2 fenc = enc*4-vec2(2);
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}



void main() {
    vec2 n = texture(text,texCoord).xy;

    vec3 normal = unpackNormal3(n);
    normal = normal * 0.5f + 0.5f;

    out_color =  vec4(normal,1);
}


