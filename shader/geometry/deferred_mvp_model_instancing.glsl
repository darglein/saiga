/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
    layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in vec4 in_data;

// instancing
layout(location = 4) in mat4 instanceModel;

#include "camera.glsl"
uniform mat4 model;

out vec3 normal;
out vec3 color;
out vec3 data;

void main()
{
    color       = in_color.xyz;
    data        = in_data.xyz;
    normal      = normalize(vec3(view * instanceModel * vec4(in_normal.xyz, 0)));
    gl_Position = viewProj * instanceModel * vec4(in_position.xyz, 1);
}



##GL_FRAGMENT_SHADER

#version 330

    uniform float userData;  // blue channel of data texture in gbuffer. Not used in lighting.

in vec3 normal;
in vec3 color;
in vec3 data;

#include "geometry_helper_fs.glsl"


void main()
{
    setGbufferData(color, normal, vec4(data.xy, userData, 0));
}
