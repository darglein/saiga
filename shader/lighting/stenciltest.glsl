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





void main() {
    gl_Position = proj * view * model * vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330


void main() {
}


