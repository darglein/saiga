/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "light_models.glsl"
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
    vec4 direction;     // xyz, w ambient intensity
    mat4 lightMatrix;
};

struct DirectionalLightData
{
    vec4 position; // xyz, w unused
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 direction; // xyz, w ambient intensity
};

// Shader storage buffers are enabled
layout (std430, binding = 2) buffer lightDataBlockPoint
{
    PointLightData pointLights[MAX_PL_COUNT];
};

layout (std430, binding = 3) buffer lightDataBlockSpot
{
    SpotLightData spotLights[MAX_SL_COUNT];
};

layout (std430, binding = 4) buffer lightDataBlockBox
{
    BoxLightData boxLights[MAX_BL_COUNT];
};

layout (std430, binding = 5) buffer lightDataBlockDirectional
{
    DirectionalLightData directionalLights[MAX_DL_COUNT];
};

layout (std140, binding = 6) uniform lightInfoBlock
{
    int pointLightCount;
    int spotLightCount;
    int boxLightCount;
    int directionalLightCount;
};

struct cluster
{
    int offset;
    int plCount;
    int slCount;
    int blCount;
};

layout (std430, binding = 7) buffer clusterBuffer
{
    int clusterX;
    int clusterY;
    int clusterZ;
    int screenWidth;
    int screenHeight;
    cluster clusterList[];
};

struct clusterItem
{
    int plIdx;
    int slIdx;
    int blIdx;
};

layout (std430, binding = 8) buffer itemBuffer
{
    clusterItem itemList[];
};

struct AssetMaterial
{
    vec4 color;
    vec4 data;
};

// TODO Paul: Fix!
float linearDepth(float d){

    float depthRange = 2.0 * d - 1.0;
    float zFar = 80.0;
    float zNear = 0.1;
    float linear = 2.0 * zNear * zFar / (zFar + zNear - depthRange * (zFar - zNear));
    return linear;
}

uint getClusterIndex(vec3 pixelCoord, int tileWidth, int tileHeight, float scale, float bias)
{
    uint zSplit       = uint(max(log2(linearDepth(pixelCoord.z)) * scale + bias, 0.0));
    uvec3 clusters    = uvec3(pixelCoord.x / tileWidth, pixelCoord.y / tileHeight, zSplit);
    uint clusterIndex = clusters.x + clusterX * clusters.y + (clusterX * clusterY) * clusters.z;
    return clusterIndex;
}

vec3 debugCluster()
{
    int tileWidth  = screenWidth / clusterX;
    int tileHeight = screenHeight / clusterY;

    float scale = float(clusterZ) / log2(80.0 / 0.1);
    float bias = float(clusterZ) * log2(0.1) / log2(80.0 / 0.1);

    uint zSplit       = uint(max(log2(linearDepth(gl_FragCoord.z)) * scale + bias, 0.0));
    uvec3 clusters    = uvec3(gl_FragCoord.x / tileWidth, gl_FragCoord.y / tileHeight, zSplit);
    uint clusterIndex = clusters.x + clusterX * clusters.y + (clusterX * clusterY) * clusters.z;
    //uint tileIndex = getClusterIndex(gl_FragCoord.xyz, tileWidth, tileHeight);
    float normLightCount = float(clusterList[clusterIndex].plCount) / 64;
    return vec3(normLightCount, 1.0 - normLightCount, 0.0);
}

vec3 calculatePointLightsClustered(AssetMaterial material, vec3 position, vec3 normal)
{
    //return debugCluster();
    vec3 result = vec3(0);
    int tileWidth  = screenWidth / clusterX;
    int tileHeight = screenHeight / clusterY;

    float scale = float(clusterZ) / log2(80.0 / 0.1);
    float bias = float(clusterZ) * log2(0.1) / log2(80.0 / 0.1);

    uint clusterIndex = getClusterIndex(gl_FragCoord.xyz, tileWidth, tileHeight, scale, bias);

    uint lightCount           = clusterList[clusterIndex].plCount;
    uint baseLightIndexOffset = clusterList[clusterIndex].offset;

    for(uint i = 0; i < lightCount; i++)
    {
        uint lightVectorIndex = itemList[baseLightIndexOffset + i].plIdx;
        PointLightData pl = pointLights[lightVectorIndex];
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
    mat4 invCameraView = inverse(view);
    vec3 result = vec3(0);
    for(int c = 0; c < boxLightCount; ++c)
    {
        BoxLightData bl = boxLights[c];
        vec4 lightColorDiffuse = bl.colorDiffuse;
        vec4 lightColorSpecular = bl.colorSpecular;

        vec4 vLight =  bl.lightMatrix * invCameraView * vec4(position,1);
        vLight = vLight / vLight.w;
        bool fragmentOutside = vLight.x<0 || vLight.x>1 || vLight.y<0 || vLight.y>1 || vLight.z<0 || vLight.z<1;
        if(fragmentOutside)
            continue;

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
