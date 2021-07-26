/**
 * Copyright (c) 2021 Darius Rückert
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


// Intensity Attenuation based on the distance to the light source.
//   - Normalized by the light radius so that we can use the same parameters for different light sizes
//   - Shifted downwards so that DistanceAttenuation(a, radius, radius) == 0
//   -> Therefore lights have a finite range and can be efficiently rendered
//
//   Used by PointLight and SpotLight
//     - Implemented in the shader light_models.glsl
float DistanceAttenuation(vec3 attenuation, float radius, float distance)
{
#if 1
    float cutoff = 1.0 / (attenuation.z * radius * radius);
    return max(0.f , 1.0 / (attenuation.z * distance * distance) - cutoff);
#else
    float x         = distance / radius;
    float cutoff    = 1.f / (attenuation[0] + attenuation[1] + attenuation[2]);
    float intensity = 1.f / (attenuation[0] + attenuation[1] * x + attenuation[2] * x * x) - cutoff;
    return max(0.f, intensity);
#endif
}

float DistanceAttenuation(vec4 attenuation_radius, float distance)
{
    return DistanceAttenuation(attenuation_radius.xyz, attenuation_radius.w, distance);
}

float spotAttenuation(vec3 fragmentLightDir, float angle, vec3 lightDir){
    float fConeCosine = angle;
    float fCosine = dot(lightDir,fragmentLightDir);
    return smoothstep(fConeCosine, (1-fConeCosine)*0.6f + fConeCosine,fCosine);
}

