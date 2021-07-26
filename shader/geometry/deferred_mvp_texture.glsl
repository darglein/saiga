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

out vec3 normal;
out vec2 texCoord;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330

uniform sampler2D image;

in vec3 normal;
in vec2 texCoord;

layout(location=0) out vec4 out_color;
layout(location=1) out vec3 out_normal;
layout(location=2) out vec3 out_position;

void main() {

     vec4 diffColor = texture(image, texCoord);

    out_color =  diffColor;
    out_normal = normalize(normal)*0.5f+0.5f;
    out_position = vertexMV;
}


