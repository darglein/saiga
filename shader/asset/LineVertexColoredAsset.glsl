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

#include "camera.glsl"
uniform mat4 model;

out vec3 v_normal;
out vec4 view_vertex;
out vec4 v_color;

void main()
{
    v_color = in_color;
     v_normal =  vec3(view * model * vec4(in_normal.xyz,0));
//    v_normal = in_normal.xyz;
    view_vertex = view * model * vec4(in_position.xyz, 1);
    gl_Position = proj * view_vertex ;
}



##GL_FRAGMENT_SHADER
#version 330
#extension GL_ARB_explicit_uniform_location : enable
uniform vec4 color = vec4(1,1,1,1);

layout(location=3) uniform int flags  = 0;

in vec4 v_color;
in vec4 view_vertex;
in vec3 v_normal;


#include "AssetFragment.glsl"

void main()
{
    vec3 n = normalize(v_normal);

    if(flags == 1)
    {
        float v = dot(n, vec3(view_vertex));
        if (v < 0) discard;
    }

    AssetMaterial material;
    material.color = v_color * color;
    material.data = vec4(0);
    render(material,vec3(0), vec3(0,1,0));
}


