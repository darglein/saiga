/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER
#version 330
#extension GL_ARB_explicit_uniform_location : enable

layout(location=0) in vec2 Position;
layout(location=1) in vec2 UV;
layout(location=2) in vec4 Color;

layout(location = 0) uniform mat4 ProjMtx;

out vec2 Frag_UV;
out vec4 Frag_Color;

void main()
{
    Frag_UV = UV;
    Frag_Color = Color;
    gl_Position = ProjMtx * vec4(Position.xy,0,1);
}

##GL_FRAGMENT_SHADER
#version 330
#extension GL_ARB_explicit_uniform_location : enable

layout(location = 1) uniform sampler2D Texture;

in vec2 Frag_UV;
in vec4 Frag_Color;

out vec4 Out_Color;

void main()
{
    Out_Color = Frag_Color * texture( Texture, Frag_UV.st);
}
