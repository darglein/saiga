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
    // lightDir = normalize(vec3(view  * vec4(model[1])));
    lightDir = normalize(vec3(view  * vec4(model[2])));
    vertexMV = vec3(view * model * vec4( in_position, 1 ));
    vertex = vec3(model * vec4( in_position, 1 ));
    gl_Position = viewProj *model* vec4(in_position,1);
}





##GL_FRAGMENT_SHADER
#version 430 core
#extension GL_ARB_explicit_uniform_location : enable
#ifdef SHADOWS
layout(location = 7) uniform sampler2DArrayShadow depthTexures;
layout(location = 8) uniform int shadow_id = 0;
#endif
layout(location = 10) uniform int active_light_id = 0;
#define ACCUMULATE


in vec3 vertexMV;
in vec3 vertex;
in vec3 lightPos;
in vec3 lightDir;

struct SpotLightData
{
    vec4 position;       // xyz, w angle
    vec4 colorDiffuse;   // rgb intensity
    vec4 colorSpecular;  // rgb specular intensity
    vec4 attenuation;    // xyz radius
    vec4 direction;      // xyzw
};

layout (std430, binding = 3) buffer lightDataBlockSpot
{
    SpotLightData spotLights[];
};


#include "lighting_helper_fs.glsl"


layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_volumetric;

#include "volumetric.glsl"

void main() {
    SpotLightData light_data = spotLights[active_light_id];

    // vec3 lightDir = light_data.direction.xyz;
    vec3 diffColor,vposition,normal,data;
    float depth;
    getGbufferData(diffColor,vposition,depth,normal,data,0);

    vec3 fragmentLightDir = normalize(lightPos-vposition);
    float intensity = light_data.colorDiffuse.w;

    float visibility = 1.0f;
#ifdef SHADOWS
     ShadowData sd = shadow_data[shadow_id];
    visibility = calculateShadowPCFArray(sd, depthTexures, shadow_id , vposition);
#endif


//    float distanceToLight = length(vposition - lightPos);
    float distanceToLight = length( dot(vposition - lightPos,lightDir) );
    float atten = spotAttenuation(fragmentLightDir,light_data.position.w,lightDir)*DistanceAttenuation(light_data.attenuation,distanceToLight);
    float localIntensity = intensity*atten*visibility; //amount of light reaching the given point

    float Idiff = localIntensity * intensityDiffuse(normal,fragmentLightDir);
    float Ispec = 0;
    if(Idiff > 0)
        Ispec = localIntensity * data.x  * intensitySpecular(vposition,normal,fragmentLightDir,40);


    vec3 color = light_data.colorDiffuse.rgb * (
                Idiff * diffColor +
                Ispec * light_data.colorSpecular.w * light_data.colorSpecular.rgb);
#ifdef VOLUMETRIC
    vec3 vf = volumetricFactorSpot(depthTexures,shadow_id, sd,vposition,vertexMV,lightPos,lightDir,light_data.position.w,light_data.attenuation) * light_data.colorDiffuse.rgb * intensity;
    out_volumetric = vec4(vf,1);
#endif
    out_color = vec4(color,1);

//    out_color = vec4(lightColor*Idiff ,Ispec); //accumulation
}


