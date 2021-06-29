/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
    layout(location = 0) in vec3 in_position;


#include "camera.glsl"
uniform mat4 model;


void main()
{
    gl_Position = vec4(in_position.x, in_position.y, 0, 1);
}



##GL_FRAGMENT_SHADER
#version 430 core

#define SINGLE_PASS_LIGHTING
#include "lighting_helper_fs.glsl"


layout(location = 0) out vec4 out_color;


void main()
{
    AssetMaterial material;
    material.color = vec4(0);
    material.data = vec4(0);
    vec3 position, normal;
    float depth;
    getGbufferData(material.color.rgb, position, depth, normal, material.data.rgb, 0);

    vec3 lighting = vec3(0);

    lighting += calculatePointLights(material, position, normal, depth);
    lighting += calculateSpotLights(material, position, normal, depth);
    lighting += calculateDirectionalLights(material, position, normal);

    out_color = vec4(lighting, 1);
}
