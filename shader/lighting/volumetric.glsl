/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


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


float volumetricFactorPoint(samplerCubeShadow shadowMap, vec3 cameraPos, vec3 fragWPos, vec3 vertexW, vec3 lightPosW, float farplane, float nearplane, vec4 attenuation, float intensity){
    const int NB_STEPS = 100;


    vec3 fragWorldPos = vertexW;
    if(distance(cameraPos,fragWPos) < distance(cameraPos,vertexW)){
        fragWorldPos = fragWPos;
//        return 1;
    }
//    return 0;

    vec3 rayVector = fragWorldPos - cameraPos;

    float rayLength = length(rayVector);
    vec3 rayDirection = rayVector / rayLength;

    float stepLength = rayLength / NB_STEPS;
//    stepLength = 0.2f;

    vec3 step = rayDirection * stepLength;

    vec3 currentPosition = vec3(cameraPos);

    float accumFog = 0;

    for (int i = 0; i < NB_STEPS; i++)
    {

        vec3 direction =  currentPosition-lightPosW;
        float visibility = 1.0f;

        float d = VectorToDepth(direction,farplane,nearplane);
        //the bias is applied with glPolygonOffset
       float shadowMapValue = texture(shadowMap, vec4(direction,d));


        float atten = getAttenuation(attenuation,distance(currentPosition,lightPosW));

        if (shadowMapValue > 0.5f)
        {
//                        accumFog += ComputeScattering(dot(rayDirection, sunDirection));
//            accumFog += dot(rayDirection, sunDirection) * dot(rayDirection, sunDirection);
            accumFog += intensity * atten;

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

float volumetricFactorSpot(sampler2DShadow shadowMap, mat4 viewToLight, vec3 fragViewPos, vec3 vertexMV, vec3 lightPos, vec3 lightDir, float angle, vec4 attenuation){
    const int NB_STEPS = 100;

    if(vertexMV.z > fragViewPos.z)
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
        vec4 shadowPos = viewToLight * vec4(currentPosition,1);
        shadowPos = shadowPos/shadowPos.w;
        //        float shadowMapValue = offset_lookup(shadowMap,shadowPos, vec2(0));


        float shadowMapValue;

        if(shadowPos.x<0 || shadowPos.x>1 || shadowPos.y<0 || shadowPos.y>1 || shadowPos.z<0 || shadowPos.z>1)
            shadowMapValue = 0;
        else
            shadowMapValue  = texture(shadowMap,shadowPos.xyz);

        vec3 fragmentLightDir = normalize(lightPos-currentPosition);
        float distanceToLight = length( dot(currentPosition - lightPos,lightDir) );
        float atten = spotAttenuation(fragmentLightDir,angle,lightDir)*getAttenuation(attenuation,distanceToLight);

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

float volumetricFactor(sampler2DShadow shadowMap, mat4 viewToLight, vec3 fragViewPos, vec3 vertexMV, vec3 sunDirection){
    const int NB_STEPS = 100;

    if(vertexMV.z > fragViewPos.z)
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
        vec4 shadowPos = viewToLight * vec4(currentPosition,1);
        shadowPos = shadowPos/shadowPos.w;
        //        float shadowMapValue = offset_lookup(shadowMap,shadowPos, vec2(0));


        float shadowMapValue;

        if(shadowPos.x<0 || shadowPos.x>1 || shadowPos.y<0 || shadowPos.y>1 || shadowPos.z<0 || shadowPos.z>1)
            shadowMapValue = 0;
        else
            shadowMapValue  = texture(shadowMap,shadowPos.xyz);


        if (shadowMapValue > 0.5f)
        {
//                        accumFog += ComputeScattering(dot(rayDirection, sunDirection));
//            accumFog += dot(rayDirection, sunDirection) * dot(rayDirection, sunDirection);
            accumFog += 1.0f;

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
