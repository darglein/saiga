/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

##GL_COMPUTE_SHADER
#version 430 core

#include "camera.glsl"

struct LightBoundingSphere
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

struct Cluster
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

// Shader storage buffers are enabled
layout (std430, binding = 10) buffer lightClusterData
{
    LightBoundingSphere clusterData[];
};

struct ClusterBounds
{
    vec3 center;
    vec3 extends;
};

layout (std430, binding = 11) buffer clusterStructures
{
    ClusterBounds cullingCluster[];
};

#define MAX_SHARED_LIGHT_SOURCES 1024

shared int visiblePlCount;
shared int visiblePls[MAX_SHARED_LIGHT_SOURCES];
shared int visibleSlCount;
shared int visibleSls[MAX_SHARED_LIGHT_SOURCES];
shared int globalOffset;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

bool testAABBIntersection(in int lightIndex, in vec3 centerAABB, in vec3 extendsAABB)
{
    vec3 c = vec3(view * vec4(clusterData[lightIndex].light.xyz, 1));
    float r = clusterData[lightIndex].light.w;

    vec3 diff = max(abs(c - centerAABB) - extendsAABB, 0);

    float sqDist = dot(diff, diff);
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
    }

    memoryBarrier();
    barrier();

    ClusterBounds bounds;
    bounds.center = cullingCluster[clusterIndex].center;
    bounds.extends = cullingCluster[clusterIndex].extends;

    for(int i = location; i < pointLightCount; i += 256)
    {
        if(testAABBIntersection(i, bounds.center, bounds.extends))
        {
            int index = atomicAdd(visiblePlCount, 1);
            if(index < MAX_SHARED_LIGHT_SOURCES)
            {
                visiblePls[index] = i;
            }
        }
    }

    for(int i = location; i < spotLightCount; i += 256)
    {
        if(testAABBIntersection(pointLightCount + i, bounds.center, bounds.extends))
        {
            int index = atomicAdd(visibleSlCount, 1);
            if(index < MAX_SHARED_LIGHT_SOURCES)
            {
                visibleSls[index] = i;
            }
        }
    }

    memoryBarrier();
    barrier();

    if(location == 0)
    {
        visiblePlCount = min(visiblePlCount, MAX_SHARED_LIGHT_SOURCES);
        visibleSlCount = min(visibleSlCount, MAX_SHARED_LIGHT_SOURCES);
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