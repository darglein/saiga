/**
 * Copyright (c) 2021 Darius Rückert
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

#define MAX_CASCADES 5
layout(location = 10) uniform int active_light_id = 0;
#ifdef SHADOWS
layout(location = 9) uniform sampler2DArrayShadow depthTexures;
layout(location = 8) uniform int cascade_offset = 0;

uniform float depthCuts[MAX_CASCADES + 1];
uniform int numCascades;
uniform float cascadeInterpolateRange = 3.0f;

void computeCascadeId(float viewDepth, int numCascades, out int cascadeId, out int interpolateCascade,
                      out float interpolateAlpha)
{
    cascadeId          = 0;
    interpolateCascade = 0;
    for (int i = 1; i <= numCascades; ++i)
    {
        if (viewDepth <= depthCuts[i])
        {
            cascadeId = i - 1;

            float mid = (depthCuts[i] + depthCuts[i - 1]) * 0.5f;
            if (viewDepth > mid && viewDepth > depthCuts[i] - cascadeInterpolateRange)
            {
                interpolateCascade = 1;
                interpolateAlpha   = (depthCuts[i] - viewDepth) / cascadeInterpolateRange;
            }
            if (viewDepth < mid && viewDepth < depthCuts[i - 1] + cascadeInterpolateRange)
            {
                interpolateCascade = 2;
                interpolateAlpha   = (viewDepth - depthCuts[i - 1]) / cascadeInterpolateRange;
            }
            break;
        }
    }

    interpolateAlpha = 1.0f - interpolateAlpha;

    if (cascadeId == 0 && interpolateCascade == 2) interpolateCascade = 0;

    if (cascadeId == numCascades - 1 && interpolateCascade == 1) interpolateCascade = 0;
}


#endif


struct DirectionalLightData
{
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 direction; // xyz, w ambient intensity
};

layout (std430, binding = 5) buffer lightDataBlockDirectional
{
    DirectionalLightData directionalLights[];
};


uniform sampler2D ssaoTex;

uniform vec3 direction;
uniform float ambientIntensity;

#include "lighting_helper_fs.glsl"

layout(location = 0) out vec4 out_color;


float getSSAOIntensity()
{
    //    ivec2 tci = ivec2(gl_FragCoord.xy);
    vec2 tc    = CalcTexCoord();
    float ssao = texture(ssaoTex, tc).r;
    return 1.0f - ssao;
    //    return 1.0f;
}



vec4 getDirectionalLightIntensity(int sampleId)
{
    DirectionalLightData light_data = directionalLights[active_light_id];

    vec3 diffColor, vposition, normal, data;
    float depth;
    getGbufferData(diffColor, vposition, depth, normal, data, sampleId);

    vec3 fragmentLightDir = direction;
    float ssao            = getSSAOIntensity();
    float intensity       = light_data.colorDiffuse.w;


    float viewDepth = -vposition.z;

    float visibility       = 1.0f;
    int cascadeId          = 0;
    int interpolateCascade = 0;
    float interpolateAlpha = 0;

#ifdef SHADOWS
    computeCascadeId(viewDepth, numCascades, cascadeId, interpolateCascade, interpolateAlpha);

    ShadowData sd = shadow_data[cascadeId + cascade_offset];
    if (interpolateCascade == 0)
    {
//        mat4 v2l = viewToLightTransforms[cascadeId];
        visibility = calculateShadowPCFArray(sd, depthTexures, cascadeId + cascade_offset, vposition);
    }
    else if (interpolateCascade == 1)
    {
        float v1 = calculateShadowPCFArray(sd, depthTexures, cascadeId+ cascade_offset, vposition);

        ShadowData sd2 = shadow_data[cascadeId + cascade_offset + 1];
        float v2 =
            calculateShadowPCFArray(sd2, depthTexures, cascadeId + 1+ cascade_offset, vposition);
        visibility = mix(v1, v2, interpolateAlpha * 0.5);
    }
    else
    {
        float v1 = calculateShadowPCFArray(sd, depthTexures, cascadeId+ cascade_offset, vposition);
        ShadowData sd2 = shadow_data[cascadeId + cascade_offset - 1];
        float v2 =
            calculateShadowPCFArray(sd2, depthTexures, cascadeId - 1+ cascade_offset, vposition);
        visibility = mix(v1, v2, interpolateAlpha * 0.5);
    }
#endif

    float localIntensity = intensity * visibility;  // amount of light reaching the given point

    float Iamb  = ssao * ambientIntensity;
    float Idiff = ssao * localIntensity * intensityDiffuse(normal, fragmentLightDir);
    float Ispec = 0;
    if (Idiff > 0) Ispec = localIntensity * data.x * intensitySpecular(vposition, normal, fragmentLightDir, 40);

    float Iemissive = data.y;
    Iemissive = 0;

    vec3 color = light_data.colorDiffuse.rgb *
                     (Idiff * diffColor + Ispec * light_data.colorSpecular.w * light_data.colorSpecular.rgb + Iamb * diffColor) +
                 Iemissive * diffColor;

    return vec4(color, 1);
}

void main()
{
    //    vec4 c = vec4(0);
    //    int s = 1;
    //    for(int i = 0 ; i < s ; ++i){
    //        c += getDirectionalLightIntensity(i);
    //    }
    //    out_color = (c / s);

    out_color = getDirectionalLightIntensity(0);
//    out_color = vec4(1,0,0,1);
}
