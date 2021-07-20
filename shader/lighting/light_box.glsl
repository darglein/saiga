/**
 * Copyright (c) 2021 Darius Rückert
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
out vec3 lightDir;



void main() {
    lightPos = vec3(view  * vec4(model[3]));
    lightDir = normalize(vec3(view  * vec4(model[2])));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER
#version 330

#ifdef SHADOWS
uniform sampler2DShadow depthTex;
#endif

in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;
in vec3 lightDir;


#include "lighting_helper_fs.glsl"
#include "volumetric.glsl"

layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_volumetric;

void main() {
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,0);

    float intensity = lightColorDiffuse.w;

    float visibility = 1.0f;
#ifdef SHADOWS
    visibility = calculateShadowPCF2(depthBiasMV,depthTex,vposition);
#endif
    //we have to this check because some fragments outside of the light volume
    //would be visible without stencilculling + depth test.
    //stencilculling + depth test must be disabled for volumetric lights
    vec4 vLight =  depthBiasMV * vec4(vposition,1);
    vLight = vLight / vLight.w;
    float fragmentInLight = 0;
    if(vLight.x>0 && vLight.x<1 && vLight.y>0 && vLight.y<1&& vLight.z>0 && vLight.z<1)
        fragmentInLight = 1;

    float localIntensity = fragmentInLight*intensity*visibility; //amount of light reaching the given point


    float Idiff = localIntensity * intensityDiffuse(normal,lightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * data.x * intensitySpecular(vposition,normal,lightDir,40);

    vec3 color = lightColorDiffuse.rgb * (
                Idiff * diffColor +
                Ispec * lightColorSpecular.w * lightColorSpecular.rgb);

#ifdef VOLUMETRIC
    vec3 vf = volumetricFactor(depthTex,depthBiasMV,vposition,vertexMV,lightDir) * lightColorDiffuse.rgb;
    out_volumetric = vec4(vf,1);
#endif
//    out_color = vec4(1);
    out_color = vec4(color,1);
}


