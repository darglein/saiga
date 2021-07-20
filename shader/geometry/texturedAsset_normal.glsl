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
layout(location=3) in vec4 in_data;


#include "camera.glsl"
uniform mat4 model;

out vec3 normal;
out vec2 texCoord;
out vec4 data;
out vec3 viewPos;

void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    data = in_data;
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    gl_Position = viewProj *model* vec4(in_position,1);
    viewPos = vec3(view * model* vec4(in_position,1));
}





##GL_FRAGMENT_SHADER

#version 330

uniform sampler2D image;
uniform float userData; //blue channel of data texture in gbuffer. Not used in lighting.

in vec3 normal;
in vec2 texCoord;
in vec4 data;
in vec3 viewPos;

#include "geometry_helper_fs.glsl"


void main() {
    vec3 color = vec3(1);

    color *= dot(normal,-normalize(viewPos));

    setGbufferData(vec3(color),normal,data);
}


