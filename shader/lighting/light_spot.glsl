/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


##GL_VERTEX_SHADER

#version 330
layout(location=0) in vec3 in_position;


#include "camera.glsl"
uniform mat4 model;

uniform vec4 position;


out vec3 vertexMV;
out vec3 vertex;
out vec3 lightPos;
out vec3 lightDir;

void main() {
    lightPos = vec3(view  * vec4(model[3]));
    lightDir = normalize(vec3(view  * vec4(model[1])));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER
#version 330

#ifdef SHADOWS
uniform sampler2DShadow depthTex;
#endif

#define ACCUMULATE

uniform vec2 shadowPlanes; //near and far plane for shadow mapping camera
uniform vec3 attenuation;
uniform vec4 position;
uniform vec3 direction;
uniform float angle;

in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;
in vec3 lightDir;


#include "lighting_helper_fs.glsl"

layout(location=0) out vec4 out_color;

float spotAttenuation(vec3 fragmentLightDir){

    float fConeCosine = angle;
    float fCosine = dot(lightDir,fragmentLightDir);
    return smoothstep(fConeCosine, (1-fConeCosine)*0.6f + fConeCosine,fCosine);

    //similar to the code above but with expensive acos functions
//     float alpha = acos(fConeCosine);
//     float beta = acos(fCosine);
//     return smoothstep(alpha,0.8f*alpha,beta);



     //old (not that good)
//          return smoothstep(fConeCosine,1,fCosine);
}


void main() {
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,0);

    vec3 fragmentLightDir = normalize(lightPos-vposition);
    float intensity = lightColorDiffuse.w;

    float visibility = 1.0f;
#ifdef SHADOWS
//    visibility = calculateShadow(depthTex,vposition);
    visibility = calculateShadowPCF2(depthBiasMV,depthTex,vposition);
//    visibility = calculateShadowPCFdither4(depthTex,vposition);
#endif


//    float distanceToLight = length(vposition - lightPos);
    float distanceToLight = length( dot(vposition - lightPos,lightDir) );
    float atten = spotAttenuation(fragmentLightDir)*getAttenuation(attenuation,distanceToLight,position.w);
    float localIntensity = intensity*atten*visibility; //amount of light reaching the given point

    float Idiff = localIntensity * intensityDiffuse(normal,fragmentLightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * data.x  * intensitySpecular(vposition,normal,fragmentLightDir,40);


    vec3 color = lightColorDiffuse.rgb * (
                Idiff * diffColor +
                Ispec * lightColorSpecular.w * lightColorSpecular.rgb);
    out_color = vec4(color,1);


//    out_color = vec4(lightColor*Idiff ,Ispec); //accumulation
}


