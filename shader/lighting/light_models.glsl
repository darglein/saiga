/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



float intensityDiffuse(vec3 normal, vec3 lightDir){
    return max(dot(normal,lightDir), 0.0);
}

float intensitySpecular(vec3 position, vec3 normal, vec3 lightDir, float exponent){
    vec3 viewDir = normalize(-position);
#if 0
    // phong shading
//    vec3 reflectDir = normalize(-reflect(lightDir,normal));
    vec3 reflectDir = -reflect(lightDir,normal);
    float specAngle = max(dot(reflectDir,viewDir),0.0);
    float i = pow(specAngle, exponent);
#else
    // blinn phong shading
    vec3 halfDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(halfDir, normal), 0.0);
    float i = pow(specAngle, exponent*4);
#endif

    return clamp(i,0.0f,1.0f);
}


//Intensity Attenuation based on the distance to the light source.
//Used by point and spot light.
float getAttenuation(vec4 attenuation, float distance){
    float radius = attenuation.w;
    //normalize the distance, so the attenuation is independent of the radius
    float x = distance / radius;
    //make sure that we return 0 if distance > radius, otherwise we would get an hard edge
    float smoothBorder = smoothstep(1.0f,0.9f,x);
    return smoothBorder / (attenuation.x +
                    attenuation.y * x +
                    attenuation.z * x * x);
}


float spotAttenuation(vec3 fragmentLightDir, float angle, vec3 lightDir){
    float fConeCosine = angle;
    float fCosine = dot(lightDir,fragmentLightDir);
    return smoothstep(fConeCosine, (1-fConeCosine)*0.6f + fConeCosine,fCosine);
}

