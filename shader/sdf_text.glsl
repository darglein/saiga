/**
 * Copyright (c) 2021 Darius Rückert
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




out vec2 texCoord;

void main() {
    texCoord = in_tex;

	gl_Position = proj * view * model * vec4(in_position.x, in_position.y, 0, 1);
}





##GL_FRAGMENT_SHADER

#version 330


uniform vec4 color = vec4(1,1,1,1);
uniform vec4 outlineColor = vec4(0,0,0,0);
uniform vec4 glowColor = vec4(0,0,0,0);

uniform vec4 outlineData = vec4(0.5f,0.5f,0.5f,0.5f);
uniform vec2 softEdgeData = vec2(0.5f,0.5f);
uniform vec2 glowData = vec2(0.5f,0.5f);

uniform float alphaMultiplier = 1.0f;

uniform sampler2D text;


in vec2 texCoord;

out vec4 out_color;


const bool SOFT_EDGES = true;
//const float SOFT_EDGE_MIN = 0.48f;
//const float SOFT_EDGE_MAX = 0.52f;

const bool OUTLINE = true;
//const float OUTLINE_MIN_VALUE0 = 0.40f;
//const float OUTLINE_MIN_VALUE1 = 0.45f;
//const float OUTLINE_MAX_VALUE0 = 0.55f;
//const float OUTLINE_MAX_VALUE1 = 0.60f;

const bool OUTER_GLOW = true;
//const vec4 OUTER_GLOW_COLOR = vec4(0,1,0,1);
//const float OUTER_GLOW_MIN_DVALUE = 0.0f;
//const float OUTER_GLOW_MAX_DVALUE = 0.5f;

void main() {
    float OUTLINE_MIN_VALUE0 = outlineData.x;
    float OUTLINE_MIN_VALUE1 = outlineData.y;
    float OUTLINE_MAX_VALUE0 = outlineData.z;
    float OUTLINE_MAX_VALUE1 = outlineData.w;

    float OUTER_GLOW_MIN_DVALUE = glowData.x;
    float OUTER_GLOW_MAX_DVALUE = glowData.y;

    float SOFT_EDGE_MIN = softEdgeData.x;
    float SOFT_EDGE_MAX = softEdgeData.y;

    vec4 baseColor = color;
    float distAlphaMask = texture(text,texCoord).r;
//    baseColor.a = distAlphaMask;

    if( SOFT_EDGES )
    {
        baseColor.a *= smoothstep ( SOFT_EDGE_MIN ,SOFT_EDGE_MAX,distAlphaMask ) ;
    }
    else
    {
        baseColor.a *= (distAlphaMask >= 0.5) ? 1.0f : 0.0f;
    }

//    baseColor = mix ( vec4(0) , baseColor , baseColor.a ) ;

    if( OUTER_GLOW )
    {
        vec4 glowc = glowColor * smoothstep(OUTER_GLOW_MIN_DVALUE, OUTER_GLOW_MAX_DVALUE, distAlphaMask) ;
        baseColor = mix ( glowc , baseColor , baseColor.a ) ;
    }else{
        baseColor = mix ( vec4(0) , baseColor , baseColor.a ) ;
    }

//    baseColor = mix ( vec4(0) , baseColor , baseColor.a ) ;

    if( OUTLINE &&
        ( distAlphaMask >= OUTLINE_MIN_VALUE0 ) &&
        ( distAlphaMask <= OUTLINE_MAX_VALUE1 ) )
    {

        float oFactor = 1.0;
        if( distAlphaMask<= OUTLINE_MIN_VALUE1 )
        {
            oFactor = smoothstep ( OUTLINE_MIN_VALUE0,OUTLINE_MIN_VALUE1,distAlphaMask ) ;
        }
        else
        {
            oFactor = smoothstep ( OUTLINE_MAX_VALUE1,OUTLINE_MAX_VALUE0,distAlphaMask ) ;

        }
        baseColor = mix ( baseColor , outlineColor, oFactor ) ;
    }

    baseColor.a *= alphaMultiplier;
    out_color = baseColor;
    return;
}


