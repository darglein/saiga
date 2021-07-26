/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;

#include "camera.glsl"
uniform mat4 model;

out vec3 normal;
out vec3 vertexMV;
out vec3 vertex;

void main() {
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330

#include "camera.glsl"
uniform mat4 model;

uniform vec4 color;

in vec3 normal;
in vec3 vertexMV;
in vec3 vertex;

layout(location=0) out vec3 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

void main() {

    vec4 diffColor = color;

    out_color =  vec3(diffColor);
}


