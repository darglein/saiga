#include "saiga/animation/animationFrame.h"

#include <glm/gtx/quaternion.hpp>
#include <iostream>

using std::cout;
using std::endl;



void AnimationNode::interpolate(const AnimationNode &other, float alpha)
{
	//cout << "node " << this->boneIndex << " " << position << " " << other.position << " " << keyFramed << endl;
    if(this->keyFramed){
        rotation = glm::slerp(rotation,other.rotation,alpha);
        rotation = glm::normalize(rotation);
        scaling = glm::mix(scaling,other.scaling,alpha);
        position = glm::mix(position,other.position,alpha);
    }
}

void AnimationNode::reset()
{
    position = vec3(0);
    rotation = quat();
    scaling = vec3(1);
    //keyFramed = false;
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






AnimationFrame::AnimationFrame()
{

}


void AnimationFrame::calculateBoneMatrices()
{
    boneMatrices.resize(boneCount);
    nodes[AnimationFrame_ROOT_NODE].traverse(mat4(), boneMatrices, nodes);
	for (unsigned int i = 0; i<boneMatrices.size(); ++i){
		boneMatrices[i] = boneMatrices[i] * boneOffsets[i];
    }
}

const std::vector<glm::mat4> &AnimationFrame::getBoneMatrices()
{
    if(boneMatrices.size() == 0)
        calculateBoneMatrices();
    return boneMatrices;
}

void AnimationFrame::setBoneMatrices(const std::vector<mat4> &value)
{
    assert(value.size() == boneCount);
    boneMatrices = value;
}


void AnimationFrame::interpolate(const AnimationFrame &k0,const AnimationFrame &k1, AnimationFrame& out, float alpha)
{
    assert(k0.boneCount == k1.boneCount);
    assert(k0.nodeCount == k1.nodeCount);
    out = k0;
    out.time = glm::mix(k0.time,k1.time,alpha);

    for (unsigned int i = 0; i<k0.nodes.size(); ++i){
        const AnimationNode& n1 = k1.nodes[i];
        out.nodes[i].interpolate(n1, alpha);
    }
    out.boneMatrices.clear(); //invalidate bone matrices
}
