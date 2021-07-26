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

void main() {
    lightPos = vec3(view  * vec4(model[3]));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER
#version 430 core
 #extension GL_ARB_explicit_uniform_location : enable
#ifdef SHADOWS
layout(location = 7) uniform samplerCubeArrayShadow cube_test;
layout(location = 8) uniform int shadow_id = 0;
uniform vec2 shadowPlanes; //near and far plane for shadow mapping camera
#endif
layout(location = 10) uniform int active_light_id = 0;



in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;

#include "lighting_helper_fs.glsl"


struct PointLightData
{
    vec4 position; // xyz, w unused
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 attenuation; // xyz radius
};

// Shader storage buffers are enabled
layout (std430, binding = 2) buffer lightDataBlockPoint
{
    PointLightData pointLights[];
};

layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_volumetric;

#include "volumetric.glsl"


void main() {

    PointLightData light_data = pointLights[active_light_id];

    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,0);

    vec3 fragmentLightDir = normalize(lightPos-vposition);
    float intensity = light_data.colorDiffuse.w;

    float visibility = 1.0f;
#ifdef SHADOWS
    ShadowData sd = shadow_data[shadow_id];
    visibility = calculateShadowCube(sd, cube_test,light_data.position.xyz,vposition, shadow_id);
#endif

    float atten = DistanceAttenuation(light_data.attenuation,distance(vposition,lightPos));
    float localIntensity = intensity*atten*visibility; //amount of light reaching the given point


    float Idiff = localIntensity * intensityDiffuse(normal,fragmentLightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * data.x  * intensitySpecular(vposition,normal,fragmentLightDir,40);


    vec3 color = light_data.colorDiffuse.rgb * (
                Idiff * diffColor +
                Ispec * light_data.colorSpecular.w * light_data.colorSpecular.rgb);
#ifdef VOLUMETRIC
    mat4 invV = inverse(view);
    vec3 camera = vec3(invV[3]);
//    vec3 fragW2 =vec3(invV * vec4(vertexMV,1));
    vec3 fragW = vec3(sd.view_to_light*vec4(vposition,1));
    vec3 lightW = light_data.position.xyz;

    vec3 vf = volumetricFactorPoint(sd, cube_test, shadow_id,camera,fragW,vertex,lightW,light_data.attenuation) * light_data.colorDiffuse.rgb * intensity;
    out_volumetric = vec4(vf,1);
#endif
    out_color = vec4(color,1);

//    out_volumetric = vec4(0);

//    out_color = vec4(lightColor*( Idiff*diffColor + Ispec*specColor),1);
//    out_color = vec4(lightColor*Idiff ,Ispec); //accumulation


}


