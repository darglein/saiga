/**
 * Copyright (c) 2017 Darius Rückert 
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
out vec3 texCoord2;
void main() {
//    gl_Position = vec4( in_position, 1 );
    texCoord = in_tex;
    normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    texCoord2 = in_position;
    gl_Position = viewProj * vec4(in_position,1);
}





##GL_FRAGMENT_SHADER

#version 330
#include "camera.glsl"
uniform mat4 model;
uniform samplerCube image;

in vec3 normal;
in vec2 texCoord;
in vec3 texCoord2;

layout(location=0) out vec3 out_color;

void main() {
     vec4 diffColor = texture(image, texCoord2);
    out_color =  vec3(diffColor);
//    out_color =  vec3(texCoord2);
//    out_color = vec4(1);
}


