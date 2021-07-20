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

#include "camera.glsl"
uniform mat4 model;

#if defined(FORWARD_LIT)
out vec3 v_position;
#endif
out vec3 v_normal;
out vec4 v_color;
out vec4 v_data;

void main()
{
    v_color = in_color;
    v_data        = in_data;
    v_normal      = normalize(vec3(view * model * vec4(in_normal.xyz, 0)));
#if defined(FORWARD_LIT)
    v_position = (view * model * vec4(in_position.xyz, 1)).rgb;
#endif
    gl_Position = viewProj * model * vec4(in_position.xyz, 1);
}



##GL_FRAGMENT_SHADER
#version 430 core

uniform vec4 color = vec4(1,1,1,1);

#if defined(FORWARD_LIT)
in vec3 v_position;
#endif
in vec3 v_normal;
in vec4 v_color;
in vec4 v_data;


#include "AssetFragment.glsl"

void main()
{
    AssetMaterial material;
    material.color = v_color * color;
    material.data = v_data;
    vec3 position = vec3(0);
#if defined(FORWARD_LIT)
    position = v_position;
#endif
    render(material, position, v_normal);
}



