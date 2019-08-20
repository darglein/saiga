/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_OPENGL
#    include "saiga/opengl/animation/boneVertex.h"
#    include "saiga/core/util/assert.h"

namespace Saiga
{
BoneVertex::BoneVertex()
{
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        boneIndices[i] = 0;
        boneWeights[i] = 0;
    }
}

void BoneVertex::addBone(int32_t index, float weight)
{
    for (int i = 0; i < MAX_BONES_PER_VERTEX; i++)
    {
        if (boneWeights[i] == 0)
        {
            boneIndices[i] = index;
            boneWeights[i] = weight;
            return;
        }
    }

    // to many weights
    SAIGA_ASSERT(0);
}

void BoneVertex::apply(const AlignedVector<mat4>& boneMatrices)
{
    mat4 boneMatrix = zeroMat4();
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        int index = (int)boneIndices[i];
        SAIGA_ASSERT(index >= 0 && index < (int)boneMatrices.size());
        boneMatrix += boneMatrices[index] * boneWeights[i];
    }

    position = boneMatrix * position;
    normal   = boneMatrix * normal;
    normal   = normalize(normal);
}

void BoneVertex::normalizeWeights()
{
    float weightSum = 0;
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        weightSum += boneWeights[i];
    }

    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        boneWeights[i] = boneWeights[i] / weightSum;
    }
}

int BoneVertex::activeBones()
{
    int count = 0;
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        if (boneWeights[i] > 0)
        {
            count++;
        }
    }
    return count;
}

template <>
void VertexBuffer<BoneVertexCD>::setVertexAttributes()
{
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glEnableVertexAttribArray(5);


    // bone indices + weights
    glVertexAttribIPointer(0, 4, GL_INT, sizeof(BoneVertexCD), (void*)(0 * sizeof(GLfloat)));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(4 * sizeof(GLfloat)));

    // position normal
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(8 * sizeof(GLfloat)));
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(12 * sizeof(GLfloat)));

    // color data
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(16 * sizeof(GLfloat)));
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(BoneVertexCD), (void*)(20 * sizeof(GLfloat)));
}

}  // namespace Saiga

#endif
