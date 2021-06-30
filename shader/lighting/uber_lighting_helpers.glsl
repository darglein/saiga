/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "light_models.glsl"
#include "camera.glsl"

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

struct DirectionalLightData
{
    vec4 colorDiffuse; // rgb intensity
    vec4 colorSpecular; // rgb specular intensity
    vec4 direction; // xyz, w ambient intensity
};

layout (std430, binding = 2) buffer lightDataBlockPoint
{
    PointLightData pointLights[];
};

layout (std430, binding = 3) buffer lightDataBlockSpot
{
    SpotLightData spotLights[];
};

layout (std430, binding = 5) buffer lightDataBlockDirectional
{
    DirectionalLightData directionalLights[];
};

layout (std140, binding = 6) uniform lightInfoBlock
{
    int pointLightCount;
    int spotLightCount;
    int directionalLightCount;

    int clusterEnabled;
};

struct Cluster
{
    int offset;
    int plCount;
    int slCount;
};

layout (std430, binding = 7) buffer clusterInfoBuffer
{
    int clusterX;
    int clusterY;
    int screenSpaceTileSize;
    int screenWidth;
    int screenHeight;
    float zNear;
    float zFar;
    float bias;
    float scale;

    int clusterListCount;
    int itemListCount;
    int tileDebug;
    int splitDebug;

    int specialNearCluster;
    float specialNearDepth;
};

layout (std430, binding = 8) buffer clusterBuffer
{
    Cluster clusterList[];
};

struct ClusterItem
{
    int lightIndex;
};

layout (std430, binding = 9) buffer itemBuffer
{
    ClusterItem itemList[];
};

struct AssetMaterial
{
    vec4 color;
    vec4 data;
};

float linearDepth(float d){

    float depthRange = 2.0 * d - 1.0;
    float linear = 2.0 * zNear * zFar / (zFar + zNear - depthRange * (zFar - zNear));
    return linear;
}

int getClusterIndex(vec2 pixelCoord, float depth)
{
    int zSplit = specialNearCluster;
    float lin = linearDepth(depth);
    if(specialNearCluster > 0 && lin < specialNearDepth)
        zSplit = 0;
    else
        zSplit += int(max(log2(lin) * scale + bias, 0.0));
    ivec3 clusters   = ivec3(pixelCoord.x / screenSpaceTileSize, pixelCoord.y / screenSpaceTileSize, zSplit);
    int clusterIndex = clusters.x + clusterX * clusters.y + (clusterX * clusterY) * clusters.z;
    return clusters.x + clusterX * clusters.y + (clusterX * clusterY) * clusters.z;
    return clusterIndex;
}

vec3 debugCluster(float depth)
{
    int clusterIndex = getClusterIndex(gl_FragCoord.xy, depth);
    float normLightCount = float(clusterList[clusterIndex].plCount + clusterList[clusterIndex].slCount) / float(tileDebug);
    if(normLightCount >= 1.0) return vec3(1);
    return vec3(normLightCount, 0.0, 1.0 - normLightCount);
}

vec3 palette[] =
{
    vec3(0,0,1),
    vec3(0,1,0),
    vec3(0,1,1),
    vec3(1,0,0),
    vec3(1,0,1),
    vec3(1,1,0),
    vec3(1,1,1),
    vec3(0,0,0)
};

vec3 debugSplits(float depth)
{
    int zSplit = specialNearCluster;
    float lin = linearDepth(depth);
    if(specialNearCluster > 0 && lin < specialNearDepth)
        zSplit = 0;
    else
        zSplit += int(max(log2(lin) * scale + bias, 0.0));
    return palette[int(mod(zSplit, 8))];
}

vec3 calculatePointLightsNoClusters(AssetMaterial material, vec3 position, vec3 normal);

vec3 calculatePointLights(AssetMaterial material, vec3 position, vec3 normal, float depth)
{
    if(clusterEnabled == 0)
        return calculatePointLightsNoClusters(material, position, normal);
    if(tileDebug > 0)
        return debugCluster(depth);
    if(splitDebug > 0)
        return debugSplits(depth);
    vec3 result = vec3(0);

    int clusterIndex = getClusterIndex(gl_FragCoord.xy, depth);

    int lightCount           = clusterList[clusterIndex].plCount;
    int baseLightIndexOffset = clusterList[clusterIndex].offset;

    for(int i = 0; i < lightCount; i++)
    {
        int lightItem = itemList[baseLightIndexOffset + i].lightIndex;
        PointLightData pl = pointLights[lightItem];
        vec3 lightPosition = (view * vec4(pl.position.xyz, 1)).rgb;
        vec4 lightColorDiffuse = pl.colorDiffuse;
        vec4 lightColorSpecular = pl.colorSpecular;
        vec4 lightAttenuation = pl.attenuation;


        vec3 fragmentLightDir = normalize(lightPosition - position);
        float intensity = lightColorDiffuse.w;

        float visibility = 1.0;

        float att = DistanceAttenuation(lightAttenuation, distance(position, lightPosition));
        float localIntensity = intensity * att * visibility;

        float Idiff = localIntensity * intensityDiffuse(normal, fragmentLightDir);
        float Ispec = 0;
        if(Idiff > 0)
            Ispec = localIntensity * material.data.x  * intensitySpecular(position, normal, fragmentLightDir, 40);
        Ispec = 0;

        vec3 color = lightColorDiffuse.rgb * (
                    Idiff * material.color.rgb +
                    Ispec * lightColorSpecular.w * lightColorSpecular.rgb);

        result += color;
    }
    return result;
}

vec3 calculateSpotLightsNoClusters(AssetMaterial material, vec3 position, vec3 normal);

vec3 calculateSpotLights(AssetMaterial material, vec3 position, vec3 normal, float depth)
{
    if(clusterEnabled == 0)
        return calculateSpotLightsNoClusters(material, position, normal);
    if(tileDebug > 0)
        return debugCluster(depth);
    if(splitDebug > 0)
        return debugSplits(depth);
    vec3 result = vec3(0);

    int clusterIndex = getClusterIndex(gl_FragCoord.xy, depth);

    int lightCount           = clusterList[clusterIndex].slCount;
    int baseLightIndexOffset = clusterList[clusterIndex].offset + clusterList[clusterIndex].plCount;

    for(int i = 0; i < lightCount; i++)
    {
        int lightItem = itemList[baseLightIndexOffset + i].lightIndex;
        SpotLightData sl = spotLights[lightItem];
        vec3 lightPosition = (view * vec4(sl.position.xyz, 1)).rgb;
        vec4 lightColorDiffuse = sl.colorDiffuse;
        vec4 lightColorSpecular = sl.colorSpecular;
        vec4 lightAttenuation = sl.attenuation;
        vec3 lightDirection = normalize((view * -sl.direction).rgb);
        float lightAngle = sl.position.w;


        vec3 fragmentLightDir = normalize(lightPosition - position);
        float intensity = lightColorDiffuse.w;

        float visibility = 1.0;

        float distanceToLight = length(dot(position - lightPosition, lightDirection));
        float att = spotAttenuation(fragmentLightDir, lightAngle, lightDirection) * DistanceAttenuation(lightAttenuation, distanceToLight);
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

vec3 calculatePointLightsNoClusters(AssetMaterial material, vec3 position, vec3 normal)
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

        float att = DistanceAttenuation(lightAttenuation, distance(position, lightPosition));
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

vec3 calculateSpotLightsNoClusters(AssetMaterial material, vec3 position, vec3 normal)
{
    vec3 result = vec3(0);
    for(int c = 0; c < spotLightCount; ++c)
    {
        SpotLightData sl = spotLights[c];
        vec3 lightPosition = (view * vec4(sl.position.xyz, 1)).rgb;
        vec4 lightColorDiffuse = sl.colorDiffuse;
        vec4 lightColorSpecular = sl.colorSpecular;
        vec4 lightAttenuation = sl.attenuation;
        vec3 lightDirection = normalize((view * -sl.direction).rgb);
        float lightAngle = sl.position.w;


        vec3 fragmentLightDir = normalize(lightPosition - position);
        float intensity = lightColorDiffuse.w;

        float visibility = 1.0;

        float distanceToLight = length(dot(position - lightPosition, lightDirection));
        float att = spotAttenuation(fragmentLightDir, lightAngle, lightDirection) * DistanceAttenuation(lightAttenuation, distanceToLight);
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
        Iemissive = 0;

        vec3 color = lightColorDiffuse.rgb *
                    (Idiff * material.color.rgb + Ispec * lightColorSpecular.w * lightColorSpecular.rgb + Iamb * material.color.rgb) +
                    Iemissive * material.color.rgb;

        result += color;
    }
    return result;
}
