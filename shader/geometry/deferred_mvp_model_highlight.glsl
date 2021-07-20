/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec3 in_color;

#include "camera.glsl"
uniform mat4 model;

out vec3 normal;
out vec3 color;

void main() {
    color = in_color;
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330


in vec3 normal;
in vec3 color;

#include "geometry_helper_fs.glsl"

void main() {

    setGbufferData(vec3(1),normal,vec4(1,0,0,0));
}


