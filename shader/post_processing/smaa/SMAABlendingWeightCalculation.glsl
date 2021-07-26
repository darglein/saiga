/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER
#version 330



#define SMAA_GLSL_3 1
#define SMAA_INCLUDE_VS 1
#define SMAA_INCLUDE_PS 0
#include "SMAA.glsl"

layout(location=0) in vec3 in_position;

out vec2 texcoord;
out vec2 pixcoord;
out vec4 offset[3];

void main()
{
  texcoord = in_position.xy * 0.5f + 0.5f;
  SMAABlendingWeightCalculationVS(texcoord,pixcoord, offset);
  gl_Position = vec4(in_position.x,in_position.y,0,1);
}

##GL_FRAGMENT_SHADER
#version 330


#define SMAA_GLSL_3
#define SMAA_INCLUDE_VS 0
#define SMAA_INCLUDE_PS 1
#include "SMAA.glsl"

uniform sampler2D edgeTex;
uniform sampler2D areaTex;
uniform sampler2D searchTex;

in vec2 texcoord;
in vec2 pixcoord;
in vec4 offset[3];


layout(location=0) out vec4 out_color;

void main()
{
    out_color = SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset,
                                                edgeTex, areaTex, searchTex, ivec4(0));
}


