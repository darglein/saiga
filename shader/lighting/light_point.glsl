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


out vec3 vertexMV;
out vec3 vertex;
out vec3 lightPos;

void main() {
    lightPos = vec3(view  * vec4(model[3]));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER
#version 330

#ifdef SHADOWS
uniform samplerCubeShadow depthTex;
#endif

uniform vec4 attenuation;

uniform vec2 shadowPlanes; //near and far plane for shadow mapping camera

in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;

#include "lighting_helper_fs.glsl"


layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_volumetric;

#include "volumetric.glsl"


void main() {
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,0);

    vec3 fragmentLightDir = normalize(lightPos-vposition);
    float intensity = lightColorDiffuse.w;

    float visibility = 1.0f;
#ifdef SHADOWS
    float farplane = shadowPlanes.x;
    float nearplane = shadowPlanes.y;
    vec3 lightW = vec3(model[3]);
    vec3 fragW = vec3(inverse(view)*vec4(vposition,1));
    visibility = calculateShadowCube(depthTex,lightW,fragW,farplane,nearplane);
   // visibility = calculateShadowCubePCF(depthTex,lightW,fragW,farplane,nearplane);
#endif

    float atten = getAttenuation(attenuation,distance(vposition,lightPos));
    float localIntensity = intensity*atten*visibility; //amount of light reaching the given point


    float Idiff = localIntensity * intensityDiffuse(normal,fragmentLightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * data.x  * intensitySpecular(vposition,normal,fragmentLightDir,40);


    vec3 color = lightColorDiffuse.rgb * (
                Idiff * diffColor +
                Ispec * lightColorSpecular.w * lightColorSpecular.rgb);
#ifdef VOLUMETRIC
    mat4 invV = inverse(view);
    vec3 camera = vec3(invV[3]);
//    vec3 fragW2 =vec3(invV * vec4(vertexMV,1));
    vec3 vf = volumetricFactorPoint(depthTex,camera,fragW,vertex,lightW,farplane,nearplane,attenuation,intensity) * lightColorDiffuse.rgb;
    out_volumetric = vec4(vf,1);
#endif
    out_color = vec4(color,1);

//    out_volumetric = vec4(0);

//    out_color = vec4(lightColor*( Idiff*diffColor + Ispec*specColor),1);
//    out_color = vec4(lightColor*Idiff ,Ispec); //accumulation


}


