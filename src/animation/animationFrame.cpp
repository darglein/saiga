#include "saiga/animation/animationFrame.h"
#include "saiga/animation/animation.h"
#include <iostream>
#include "saiga/util/assert.h"

using std::cout;
using std::endl;



AnimationNode::AnimationNode(const AnimationNode &n0, const AnimationNode &n1, float alpha)
{
    name = n0.name;
    children = n0.children;
    index = n0.index;
    boneIndex = n0.boneIndex;
    keyFramed = n0.keyFramed;

    if(this->keyFramed){
        rotation = glm::slerp(n0.rotation,n1.rotation,alpha);
        rotation = glm::normalize(rotation);
        scaling = glm::mix(n0.scaling,n1.scaling,alpha);
        position = glm::mix(n0.position,n1.position,alpha);
    }else{
        matrix = n0.matrix;
    }
}


void AnimationNode::reset()
{
    position = vec4(0);
    rotation = quat();
    scaling = vec4(1);
}



void AnimationNode::traverse(mat4 m, std::vector<mat4> &out_boneMatrices, std::vector<AnimationNode> &nodes)
{
    if(keyFramed){
        matrix = createTRSmatrix(position,rotation,scaling);
    }

    m = m*matrix;

    if(boneIndex != -1){
        out_boneMatrices[boneIndex] = m;
    }


	for (int i : children){
		nodes[i].traverse(m, out_boneMatrices, nodes);
	}
}



AnimationFrame::AnimationFrame(const AnimationFrame &k0, const AnimationFrame &k1, float alpha)
{
    assert(k0.nodeCount == k1.nodeCount);

    if(alpha == 0){
        *this = k0;
        return;
    }else if(alpha == 1){
        *this = k1;
        return;
    }

    //copy everything except the bonematrices
    time = (1-alpha) * k0.time + alpha * k1.time;

    nodeCount = k0.nodeCount;
    nodes.reserve(nodeCount);
    for (unsigned int i = 0; i<k0.nodes.size(); ++i){
        nodes.emplace_back(k0.nodes[i],k1.nodes[i],alpha);
    }
}


void AnimationFrame::calculateBoneMatrices(const Animation &parent)
{
    boneMatrices.resize(parent.boneCount);
    nodes[0].traverse(mat4(1), boneMatrices, nodes);
	for (unsigned int i = 0; i<boneMatrices.size(); ++i){
        boneMatrices[i] = boneMatrices[i] * parent.boneOffsets[i];
    }
}

const std::vector<mat4> &AnimationFrame::getBoneMatrices(const Animation &parent)
{
    if(boneMatrices.size() == 0)
        calculateBoneMatrices(parent);
    return boneMatrices;
}

void AnimationFrame::setBoneMatrices(const std::vector<mat4> &value)
{
//    assert(value.size() == boneCount);
    boneMatrices = value;
}



