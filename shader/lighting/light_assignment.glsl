/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430 core

#include "camera.glsl"

struct LightClusterData
{
    vec4 light; // xyz, w radius
};

layout (std140, binding = 6) uniform lightInfoBlock
{
    int pointLightCount;
    int spotLightCount;
    int directionalLightCount;

    int clusterEnabled;
};

struct cluster
{
    int offset;
    int plCount;
    int slCount;
};

layout (std430, binding = 7) coherent buffer clusterInfoBuffer
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
};

layout (std430, binding = 8) buffer clusterBuffer
{
    cluster clusterList[];
};

struct clusterItem
{
    int lightIndex;
};

layout (std430, binding = 9) buffer itemBuffer
{
    clusterItem itemList[];
};

// Shader storage buffers are enabled
layout (std430, binding = 10) buffer lightClusterData
{
    LightClusterData clusterData[];
};

struct clusterBounds
{
    vec3 minB;
    vec3 maxB;
};

layout (std430, binding = 11) buffer clusterStructures
{
    clusterBounds cullingCluster[];
};


shared int visiblePlCount;
shared int visiblePls[1024];
shared int visibleSlCount;
shared int visibleSls[1024];
shared int globalOffset;
shared clusterBounds bounds;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

bool testAABBIntersection(in int lightIndex, in vec3 minB, in vec3 maxB)
{
    vec3 c = vec3(view * vec4(clusterData[lightIndex].light.xyz, 1));
    float r = clusterData[lightIndex].light.w;

    float sqDist = 0.0;
    for (int i = 0; i < 3; i++)
    {
        if (c[i] < minB[i]) sqDist += (minB[i] - c[i]) * (minB[i] - c[i]);
        if (c[i] > maxB[i]) sqDist += (c[i] - maxB[i]) * (c[i] - maxB[i]);
    }

    float rSqared = r * r;
    return (sqDist <= rSqared);
}

void main()
{
    int location = int(gl_LocalInvocationIndex);
    ivec3 clusterID = ivec3(gl_WorkGroupID.xyz);
    ivec3 workGroupSize = ivec3(gl_WorkGroupSize.xyz);
    int clusterIndex = clusterID.x + clusterX * clusterID.y + (clusterX * clusterY) * clusterID.z;

    if(location == 0)
    {
        visiblePlCount = 0;
        visibleSlCount = 0;
        bounds.minB = cullingCluster[clusterIndex].minB;
        bounds.maxB = cullingCluster[clusterIndex].maxB;
    }

    memoryBarrier();
    barrier();

    for(int i = location; i < pointLightCount; i += 256)
    {
        if(testAABBIntersection(i, bounds.minB, bounds.maxB))
        {
            int index = atomicAdd(visiblePlCount, 1);
            if(index < 1024)
            {
                visiblePls[index] = i;
            }
        }
    }

    for(int i = location; i < spotLightCount; i += 256)
    {
        if(testAABBIntersection(pointLightCount + i, bounds.minB, bounds.maxB))
        {
            int index = atomicAdd(visibleSlCount, 1);
            if(index < 1024)
            {
                visibleSls[index] = i;
            }
        }
    }

    memoryBarrier();
    barrier();

    if(location == 0)
    {
        visiblePlCount = min(visiblePlCount, 1024);
        visibleSlCount = min(visibleSlCount, 1024);
        globalOffset = atomicAdd(itemListCount, visiblePlCount + visibleSlCount);
        clusterList[clusterIndex].offset = globalOffset;
        clusterList[clusterIndex].plCount = visiblePlCount;
        clusterList[clusterIndex].slCount = visibleSlCount;
    }

    memoryBarrier();
    barrier();

    for(int i = location; i < visiblePlCount; i += 256)
    {
        itemList[globalOffset + i].lightIndex = visiblePls[i];
    }

    for(int i = location; i < visibleSlCount; i += 256)
    {
        itemList[globalOffset + visiblePlCount + i].lightIndex = visibleSls[i];
    }
}