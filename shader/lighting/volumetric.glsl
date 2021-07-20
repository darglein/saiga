/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "geometry/intersection.glsl"

uniform float volumetricDensity = 0.01f;

// Mie scaterring approximated with Henyey-Greenstein phase function.
float ComputeScattering(float lightDotView)
{
    float G_SCATTERING = 0.1f;
    float PI = 3.0f;
    float result = 1.0f - G_SCATTERING * G_SCATTERING;
    result /= (4.0f * PI * pow(1.0f + G_SCATTERING * G_SCATTERING - (2.0f * G_SCATTERING) *  lightDotView, 1.5f));
    return result;
}

float volumetricFactorPoint(ShadowData sd, samplerCubeArrayShadow shadowMap, int layer, vec3 cameraPos, vec3 fragWPos, vec3 vertexW, vec3 lightPosW, vec4 attenuation){
    const int NB_STEPS = 10;

    float farplane = sd.shadow_planes.x;
    float nearplane = sd.shadow_planes.y;

    vec3 rayOrigin = cameraPos;

    vec3 rayEnd = vertexW;
    rayEnd = (distance(rayOrigin, fragWPos) < distance(rayOrigin, vertexW)) ? fragWPos : vertexW;


    vec3 rayVector = rayEnd - rayOrigin;
    float rayLength = length(rayVector);
    vec3 rayDirection = rayVector / rayLength;

    #if 1
    //clamp ray start to intersection with sphere
    float t1, t2;
    RaySphere(rayOrigin, rayDirection, lightPosW, attenuation.w, t1, t2);
    if (t1 > 0){
        rayOrigin = rayOrigin + rayDirection * t1;
    }
    rayVector = rayEnd - rayOrigin;
    rayLength = length(rayVector);
    rayDirection = rayVector / rayLength;
    #endif

    float stepLength = rayLength / NB_STEPS;
    vec3 step = rayDirection * stepLength;

    const float dither[16] = float[](0.0f, 0.5f, 0.125f, 0.625f,
    0.75f, 0.22f, 0.875f, 0.375f,
    0.1875f, 0.6875f, 0.0625f, 0.5625,
    0.9375f, 0.4375f, 0.8125f, 0.3125);
    ivec2 tci = ivec2(gl_FragCoord.xy);
    int ditherId = tci.y % 4 * 4 + tci.x % 4;
    //     int ditherId = 0;

    //    vec3 currentPosition = rayOrigin + step * (dither[ditherId] * 0.5f - 0.5f);
    vec3 currentPosition = rayOrigin + step * (dither[ditherId]);

    float accumFog = 0;

    for (int i = 0; i < NB_STEPS; i++)
    {

        vec3 direction =  currentPosition-lightPosW;
        float visibility = 1.0f;

        float d = VectorToDepth(direction, farplane, nearplane);
        //the bias is applied with glPolygonOffset
        float shadowMapValue = texture(shadowMap, vec4(direction, layer), d);


        float atten = DistanceAttenuation(attenuation, distance(currentPosition, lightPosW));

        if (shadowMapValue > 0.5f)
        {
            //                        accumFog += ComputeScattering(dot(rayDirection, sunDirection));
            //            accumFog += dot(rayDirection, sunDirection) * dot(rayDirection, sunDirection);
            accumFog += 1.0 * atten;

        }
        currentPosition += step;
    }
    accumFog /= NB_STEPS;

    float tau = volumetricDensity;
    accumFog = (-exp(-rayLength*tau*accumFog)+1);
    //    accumFog = (-exp(-rayLength*tau)+1) * accumFog;

    //    return 1;
    return accumFog;
}

float volumetricFactorSpot(sampler2DArrayShadow shadowMap, int shadow_id,
ShadowData sd, vec3 fragViewPos, vec3 vertexMV, vec3 lightPos, vec3 lightDir, float angle, vec4 attenuation)
{
    const int NB_STEPS = 100;

    if (vertexMV.z > fragViewPos.z)
    fragViewPos = vertexMV;
    vec3 rayVector = fragViewPos;

    float rayLength = length(rayVector);
    vec3 rayDirection = rayVector / rayLength;

    float stepLength = rayLength / NB_STEPS;
    //    stepLength = 0.2f;

    vec3 step = rayDirection * stepLength;

    vec3 currentPosition = vec3(0);

    float accumFog = 0;

    for (int i = 0; i < NB_STEPS; i++)
    {
        vec4 shadowPos = sd.view_to_light * vec4(currentPosition, 1);
        shadowPos = shadowPos/shadowPos.w;
        //        float shadowMapValue = offset_lookup(shadowMap,shadowPos, vec2(0));


        float shadowMapValue;

        if (shadowPos.x<0 || shadowPos.x>1 || shadowPos.y<0 || shadowPos.y>1 || shadowPos.z<0 || shadowPos.z>1)
        shadowMapValue = 0;
        else
        shadowMapValue  = texture(shadowMap, vec4(shadowPos.x, shadowPos.y, shadow_id, shadowPos.z));

        vec3 fragmentLightDir = normalize(lightPos-currentPosition);
        float distanceToLight = length(dot(currentPosition - lightPos, lightDir));
        float atten = spotAttenuation(fragmentLightDir, angle, lightDir) * DistanceAttenuation(attenuation, distanceToLight);

        if (shadowMapValue > 0.5f)
        {
            //                        accumFog += ComputeScattering(dot(rayDirection, sunDirection));
            //            accumFog += dot(rayDirection, sunDirection) * dot(rayDirection, sunDirection);
            accumFog += 1.0f * atten;

        }
        currentPosition += step;
    }
    accumFog /= NB_STEPS;

    float tau = volumetricDensity;
    accumFog = (-exp(-rayLength*tau*accumFog)+1);
    //    accumFog = (-exp(-rayLength*tau)+1) * accumFog;

    //    return 0;
    return accumFog;
}

