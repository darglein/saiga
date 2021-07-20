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

uniform vec4 position;


out vec3 vertexMV;
out vec3 vertex;
out vec3 lightPos;

void main() {
    gl_Position = viewProj *model* vec4(in_position, 1);
}





##GL_FRAGMENT_SHADER

#version 330

#include "camera.glsl"
uniform mat4 model;

uniform sampler2D deferred_diffuse;
uniform sampler2D deferred_normal;
uniform sampler2D deferred_depth;
uniform sampler2D deferred_position;
uniform vec2 screen_size;

uniform vec4 color;
uniform vec3 attenuation;
uniform vec4 position;

in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;

layout(location=0) out vec4 out_color;


void main() {
    out_color = color;
}


