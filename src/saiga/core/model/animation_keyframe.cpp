/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "animation_keyframe.h"

#include "saiga/core/util/assert.h"

#include "animation.h"

namespace Saiga
{
AnimationNode::AnimationNode(const AnimationNode& n0, const AnimationNode& n1, float alpha)
{
    name      = n0.name;
    children  = n0.children;
    index     = n0.index;
    boneIndex = n0.boneIndex;
    keyFramed = n0.keyFramed;

    if (this->keyFramed)
    {
        rotation = slerp(n0.rotation, n1.rotation, alpha);
        rotation = normalize(rotation);
        scaling  = mix(n0.scaling, n1.scaling, alpha);
        position = mix(n0.position, n1.position, alpha);
    }
    else
    {
        matrix = n0.matrix;
    }
}


void AnimationNode::reset()
{
    position = make_vec4(0);
    rotation = quat::Identity();
    scaling  = make_vec4(1);
}



void AnimationNode::traverse(mat4 m, AlignedVector<mat4>& out_boneMatrices, std::vector<AnimationNode>& nodes)
{
    if (keyFramed)
    {
        matrix = createTRSmatrix(position, rotation, scaling);
    }

    m = m * matrix;

    if (boneIndex != -1)
    {
        out_boneMatrices[boneIndex] = m;
    }


    for (int i : children)
    {
        nodes[i].traverse(m, out_boneMatrices, nodes);
    }
}



AnimationKeyframe::AnimationKeyframe(const AnimationKeyframe& k0, const AnimationKeyframe& k1, float alpha)
{
    SAIGA_ASSERT(k0.nodeCount == k1.nodeCount);

    if (alpha == 0)
    {
        *this = k0;
        return;
    }
    else if (alpha == 1)
    {
        *this = k1;
        return;
    }

    // copy everything except the bonematrices
    time = (1 - alpha) * k0.time + alpha * k1.time;

    nodeCount = k0.nodeCount;
    nodes.reserve(nodeCount);
    for (unsigned int i = 0; i < k0.nodes.size(); ++i)
    {
        nodes.emplace_back(k0.nodes[i], k1.nodes[i], alpha);
    }
}


void AnimationKeyframe::calculateBoneMatrices(const Animation& parent)
{
    boneMatrices.resize(parent.boneCount);
    nodes[0].traverse(mat4::Identity(), boneMatrices, nodes);
    for (unsigned int i = 0; i < boneMatrices.size(); ++i)
    {
        boneMatrices[i] = boneMatrices[i] * parent.boneOffsets[i];
    }
}

const AlignedVector<mat4>& AnimationKeyframe::getBoneMatrices(const Animation& parent)
{
    if (boneMatrices.size() == 0) calculateBoneMatrices(parent);
    return boneMatrices;
}

void AnimationKeyframe::setBoneMatrices(const AlignedVector<mat4>& value)
{
    //    SAIGA_ASSERT(value.size() == boneCount);
    boneMatrices = value;
}

}  // namespace Saiga
