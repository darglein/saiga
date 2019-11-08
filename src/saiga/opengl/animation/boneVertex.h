/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/Align.h"

#include <saiga/opengl/vertexBuffer.h>

namespace Saiga
{
/**
 * We are using a maximum number of 4 bones per vertex here, because it fits nicely in a vec4 on the gpu
 * and was sufficient in all cases I have encountered so far.
 */

#define MAX_BONES_PER_VERTEX 4

struct SAIGA_OPENGL_API BoneVertex
{
   public:
    int32_t boneIndices[MAX_BONES_PER_VERTEX];
    float boneWeights[MAX_BONES_PER_VERTEX];

    vec4 position;
    vec4 normal;

    BoneVertex();

    // add a bone with given index and weight to this vertex
    void addBone(int32_t index, float weight);

    // applies an array of bonematrices to this position and normal.
    // That is a copy of the vertex shader functionality.
    void apply(const AlignedVector<mat4>& boneMatrices);

    // normalizes the weights so that the sum is 1.
    void normalizeWeights();

    // number of bones with weight > 0
    int activeBones();
};


struct SAIGA_OPENGL_API BoneVertexT : public BoneVertex
{
   public:
    vec2 texture;
};

struct SAIGA_OPENGL_API BoneVertexCD : public BoneVertex
{
   public:
    vec4 color;
    vec4 data;
};


template <>
SAIGA_OPENGL_API void VertexBuffer<BoneVertexCD>::setVertexAttributes();

}  // namespace Saiga
