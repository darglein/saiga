/**
 * Copyright (c) 2020 Paul Himmler
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

out vec3 v_position;
out vec3 v_normal;
out vec2 v_texCoord;
out vec4 v_data;

void main() {
    v_texCoord = in_tex;
    v_data = in_data;
    v_normal = normalize(vec3(view*model * vec4( in_normal, 0 )));
    v_position = (view * model * vec4(in_position,0)).rgb;
    gl_Position = viewProj * model * vec4(in_position,1);
}


##GL_FRAGMENT_SHADER
#version 330

uniform sampler2D image;
uniform float userData; //blue channel of data texture in gbuffer. Not used in lighting.

in vec3 v_position;
in vec3 v_normal;
in vec2 v_texCoord;
in vec4 v_data;


#include "standardAssetFragment.glsl"

void main()
{
    AssetMaterial material;
    material.color =texture(image, v_texCoord);
    material.data = v_data;
    render(material, v_position, v_normal);
}



