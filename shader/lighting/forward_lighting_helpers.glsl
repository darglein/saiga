/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "lighting/light_models.glsl"
#include "camera.glsl"

#ifndef MAX_PL_COUNT
#define MAX_PL_COUNT 256
#endif
#ifndef MAX_SL_COUNT
#define MAX_SL_COUNT 256
#endif
#ifndef MAX_BL_COUNT
#define MAX_BL_COUNT 256
#endif
#ifndef MAX_DL_COUNT
#define MAX_DL_COUNT 256
#endif

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
    vec4 colorDiffuse;  // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 min_w;         // xyz, w unused
    vec4 max_w;         // xyz, w unused
    vec4 direction;     // xyz, w ambient intensity
};

struct DirectionalLightData
{
    vec4 position; // xyz, w unused
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 direction; // xyz, w ambient intensity
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
    DirectionalLightData directionalLights[MAX_DL_COUNT];
};

layout (std140) uniform lightInfoBlock
{
    int pointLightCount;
    int spotLightCount;
    int boxLightCount;
    int directionalLightCount;
};

struct AssetMaterial
{
    vec4 color;
    vec4 data;
};

vec3 calculatePointLights(AssetMaterial material, vec3 position, vec3 normal)
{
    vec3 result = vec3(0);
    for(int c = 0; c < pointLightCount; ++c)
    {
        PointLightData pl = pointLights[c];
        vec3 lightPosition = (view * vec4(pl.position.xyz, 1)).rgb;
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
        vec3 lightPosition = (view * vec4(sl.position.xyz, 1)).rgb;
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

vec3 calculateBoxLights(AssetMaterial material, vec3 position, vec3 normal)
{
    vec3 result = vec3(0);
    for(int c = 0; c < boxLightCount; ++c)
    {
        BoxLightData bl = boxLights[c];
        vec4 lightColorDiffuse = bl.colorDiffuse;
        vec4 lightColorSpecular = bl.colorSpecular;

        vec3 bounds[2];

        bounds[0] = (view * bl.min_w).rgb;
        bounds[1] = (view * bl.max_w).rgb;

        if(any(lessThan(position, bounds[0])) || any(greaterThan(position, bounds[1])))
        {
            continue;
        }
        //result = vec3(0.5);
        //continue;

        vec3 fragmentLightDir = normalize((view * vec4(bl.direction.rgb, 0.0)).rgb);
        float intensity       = lightColorDiffuse.w;
        float visibility      = 1.0;
        float localIntensity  = intensity * visibility;

        float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
        float Ispec = 0;
        if (Idiff > 0) Ispec = localIntensity * material.data.x * intensitySpecular(position, normal, fragmentLightDir, 40);

        vec3 color = lightColorDiffuse.rgb * (
                    Idiff * material.color.rgb +
                    Ispec * lightColorSpecular.w * lightColorSpecular.rgb);

        result += color;
    }
    return result;
}

vec3 calculateDirectionalLights(AssetMaterial material, vec3 position, vec3 normal)
{
    vec3 result = vec3(0);
    for(int c = 0; c < directionalLightCount; ++c)
    {
        DirectionalLightData dl = directionalLights[c];
        vec4 lightColorDiffuse = dl.colorDiffuse;
        vec4 lightColorSpecular = dl.colorSpecular;
        float ambientIntensity = dl.direction.w;

        vec3 fragmentLightDir = normalize((view * vec4(-dl.direction.rgb, 0.0)).rgb);
        float intensity       = lightColorDiffuse.w;
        float visibility      = 1.0;
        float localIntensity  = intensity * visibility;

        float Iamb  = ambientIntensity;
        float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
        float Ispec = 0;
        if (Idiff > 0) Ispec = localIntensity * material.data.x * intensitySpecular(position, normal, fragmentLightDir, 40);

        float Iemissive = material.data.y;

        vec3 color = lightColorDiffuse.rgb *
                    (Idiff * material.color.rgb + Ispec * lightColorSpecular.w * lightColorSpecular.rgb + Iamb * material.color.rgb) +
                    Iemissive * material.color.rgb;

        result += color;
    }
    return result;
}
