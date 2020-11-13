/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

struct AssetMaterial
{
    vec4 color;
    vec4 data;
};

#if defined(DEFERRED)
#include "geometry/geometry_helper_fs.glsl"
#elif defined(DEPTH)
#else
layout(location=0) out vec4 out_color;
#include "lighting/light_models.glsl"
#include "camera.glsl"
vec3 calculatePointLights(AssetMaterial material, vec3 position, vec3 normal);
vec3 calculateSpotLights(AssetMaterial material, vec3 position, vec3 normal);
#endif


#define MAX_PL_COUNT 1024
#define MAX_SL_COUNT 750
#define MAX_BL_COUNT 1024
#define MAX_DL_COUNT 1024

struct PointLightData
{
    vec4 position; // xyz, w unused
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 attenuation; // xyz radius
};

struct SpotLightData
{
    vec4 position;       // xyz, w angle
    vec4 colorDiffuse;   // rgb intensity
    vec4 colorSpecular;  // rgb specular intensity
    vec4 attenuation;    // xyz radius
    vec4 direction;      // xyzw
};

struct BoxLightData
{
    vec4 position; // xyz, w unused
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
};

struct DirectionalLightData
{
    vec4 position; // xyz, w unused
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 direction; // xyz, w unused
};

layout (std140) uniform lightDataBlockPoint
{
    PointLightData pointLights[MAX_PL_COUNT];
};

layout (std140) uniform lightDataBlockSpot
{
    SpotLightData spotLights[MAX_SL_COUNT];
};

layout (std140) uniform lightDataBlockBox
{
    BoxLightData boxLights[MAX_BL_COUNT];
};

layout (std140) uniform lightDataBlockDirectional
{
    DirectionalLightData directionaLLights[MAX_DL_COUNT];
};

layout (std140) uniform lightInfoBlock
{
    int pointLightCount;
    int spotLightCount;
    int boxLightCount;
    int directionalLightCount;
};

void render(AssetMaterial material, vec3 position, vec3 normal)
{
#if defined(DEFERRED)
    setGbufferData(vec3(material.color), normal, material.data);
#elif defined(DEPTH)
#else
    vec3 lighting = vec3(0);

    lighting += calculatePointLights(material, position, normal);
    lighting += calculateSpotLights(material, position, normal);

    float Iamb = 0.02;
    lighting += Iamb * material.color.rgb;

    out_color = vec4(lighting, 1);
#endif
}

#if defined(DEFERRED)
#elif defined(DEPTH)
#else
vec3 calculatePointLights(AssetMaterial material, vec3 position, vec3 normal)
{
    vec3 result = vec3(0);
    for(int c = 0; c < pointLightCount; ++c)
    {
        PointLightData pl = pointLights[c];
        vec3 lightPosition = (view * vec4(pl.position.xyz, 0.0)).rgb;
        vec4 lightColorDiffuse = pl.colorDiffuse;
        vec4 lightColorSpecular = pl.colorSpecular;
        vec4 lightAttenuation = pl.attenuation;


        vec3 fragmentLightDir = normalize(lightPosition - position);
        float intensity = lightColorDiffuse.w;

        float visibility = 1.0;

        float att = getAttenuation(lightAttenuation, distance(position, lightPosition));
        float localIntensity = intensity * att * visibility;

        float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
        float Ispec = 0;
        if(Idiff > 0)
            Ispec = localIntensity * material.data.x  * intensitySpecular(position, normal, fragmentLightDir, 40);


        vec3 color = lightColorDiffuse.rgb * (
                    Idiff * material.color.rgb +
                    Ispec * lightColorSpecular.w * lightColorSpecular.rgb);

        result += color;
    }
    return result;
}


vec3 calculateSpotLights(AssetMaterial material, vec3 position, vec3 normal)
{
    vec3 result = vec3(0);
    for(int c = 0; c < spotLightCount; ++c)
    {
        SpotLightData sl = spotLights[c];
        vec3 lightPosition = (view * vec4(sl.position.xyz, 0.0)).rgb;
        vec4 lightColorDiffuse = sl.colorDiffuse;
        vec4 lightColorSpecular = sl.colorSpecular;
        vec4 lightAttenuation = sl.attenuation;
        vec3 lightDirection = normalize((view * sl.direction).rgb);
        float lightAngle = sl.position.w;


        vec3 fragmentLightDir = normalize(lightPosition - position);
        float intensity = lightColorDiffuse.w;

        float visibility = 1.0;

        float distanceToLight = length(dot(position - lightPosition, lightDirection));
        float att = spotAttenuation(fragmentLightDir, lightAngle, lightDirection) * getAttenuation(lightAttenuation, distanceToLight);
        float localIntensity = intensity * att * visibility;

        float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
        float Ispec = 0;
        if(Idiff > 0)
            Ispec = localIntensity * material.data.x  * intensitySpecular(position, normal, fragmentLightDir, 40);


        vec3 color = lightColorDiffuse.rgb * (
                    Idiff * material.color.rgb +
                    Ispec * lightColorSpecular.w * lightColorSpecular.rgb);

        result += color;
    }
    return result;
}
#endif
