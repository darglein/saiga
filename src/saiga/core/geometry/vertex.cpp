/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "vertex.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"


namespace Saiga
{
bool Vertex::operator==(const Vertex& other) const
{
    return position == other.position;
}

std::ostream& operator<<(std::ostream& os, const Vertex& vert)
{
    os << vert.position.transpose();
    return os;
}

bool VertexN::operator==(const VertexN& other) const
{
    return Vertex::operator==(other) && normal == other.normal;
}

std::ostream& operator<<(std::ostream& os, const VertexN& vert)
{
    os << "VertexN" << vert.position.transpose() << ",";
    os << vert.normal.transpose();
    return os;
}

bool VertexNT::operator==(const VertexNT& other) const
{
    return VertexN::operator==(other) && texture == other.texture;
}

std::ostream& operator<<(std::ostream& os, const VertexNT& vert)
{
    os << "VertexNT" << vert.position.transpose() << ",";
    os << vert.normal.transpose() << ",";
    os << vert.texture.transpose();
    return os;
}


bool VertexNC::operator==(const VertexNC& other) const
{
    return VertexN::operator==(other) && color == other.color && data == other.data;
}

std::ostream& operator<<(std::ostream& os, const VertexNC& vert)
{
    os << "VertexNC" << vert.position.transpose() << ",";
    os << vert.normal.transpose() << ",";
    os << vert.color.transpose() << ",";
    os << vert.data.transpose();
    return os;
}


void BoneInfo::addBone(int32_t index, float weight)
{
    for (int i = 0; i < MAX_BONES_PER_VERTEX; i++)
    {
        if (bone_weights[i] == 0)
        {
            bone_indices[i] = index;
            bone_weights[i] = weight;
            return;
        }
    }

    // to many weights
    SAIGA_ASSERT(0);
}


void BoneInfo::normalizeWeights()
{
    float weightSum = 0;
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        weightSum += bone_weights[i];
    }

    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        bone_weights[i] = bone_weights[i] / weightSum;
    }
}

int BoneInfo::activeBones()
{
    int count = 0;
    for (int i = 0; i < MAX_BONES_PER_VERTEX; ++i)
    {
        if (bone_weights[i] > 0)
        {
            count++;
        }
    }
    return count;
}


}  // namespace Saiga
