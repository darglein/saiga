/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec4 in_position;


#include "camera.glsl"
uniform mat4 model;

out vec3 normal;
out vec3 vertexMV;
out vec3 vertex;

void main() {
    gl_Position = viewProj *model* vec4(in_position.xyz,1);
}





##GL_FRAGMENT_SHADER

#version 330

#include "camera.glsl"
uniform mat4 model;

uniform vec4 color = vec4(0,1,0,1);

in vec3 normal;
in vec3 vertexMV;
in vec3 vertex;

layout(location=0) out vec4 out_color;

void main()
{
   out_color =   color;
}


