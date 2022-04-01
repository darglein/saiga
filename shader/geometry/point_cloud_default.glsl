/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER
#version 330
#extension GL_ARB_explicit_uniform_location : enable
layout(location=0) in vec3 in_position;
layout(location=1) in vec3 in_normal;
layout(location=2) in vec4 in_color;

#include "camera.glsl"
uniform mat4 model;

out vec3 color;
out vec3 normal;

void main() {
    color = in_color.rgb;
    gl_Position = viewProj *model* vec4(in_position.xyz,1);
    normal = (view*model* vec4(in_normal.xyz,0)).xyz;
}





##GL_FRAGMENT_SHADER
#version 330
#extension GL_ARB_explicit_uniform_location : enable

in vec3 color;
in vec3 normal;
#include "geometry/geometry_helper_fs.glsl"


layout(location=1) uniform int cull_backface;

void main() {
//    out_color = vec4(color,1);


    if(cull_backface != 0 && dot(normal, vec3(0,0,1)) < 0) discard;

    #ifdef SHADOW
    float z = gl_FragCoord.z;
    z += 0.01f;

    gl_FragDepth = z;
    #endif

    vec3 data = vec3(0, 0, 0);
    setGbufferData(vec3(color), normalize(normal), vec4(data.xy, 0, 0));
}


