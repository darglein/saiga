/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER
#version 330

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_color;

#include "camera.glsl"
uniform mat4 model;

out vec4 v_color;

void main()
{
    v_color = in_color;
    gl_Position = viewProj * model * vec4(in_position.xyz, 1);
}



##GL_FRAGMENT_SHADER
#version 330

uniform vec4 color = vec4(1,1,1,1);

in vec4 v_color;


#include "AssetFragment.glsl"

void main()
{
    AssetMaterial material;
    material.color = v_color * color;
    material.data = vec4(0);
    render(material,vec3(0));
}


