#include "saiga/animation/animationFrame.h"

#include <glm/gtx/quaternion.hpp>
#include <iostream>

using std::cout;
using std::endl;



void AnimationNode::interpolate(AnimationNode &other, float alpha)
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

AnimationFrame::AnimationFrame(const AnimationFrame &other)
{
	*this = other;
}


void AnimationFrame::calculateFromTree()
{
	nodes[rootNode].traverse(mat4(), boneMatrices, nodes);
	for (unsigned int i = 0; i<boneMatrices.size(); ++i){
		boneMatrices[i] = boneMatrices[i] * boneOffsets[i];
	}
}

void AnimationFrame::interpolate(AnimationFrame &k0, AnimationFrame &k1, AnimationFrame& out, float alpha)
{
	//cout << "AnimationFrame::interpolate " << alpha << endl;
	out = k0;

    for (unsigned int i = 0; i<k0.nodes.size(); ++i){
		AnimationNode& n1 = k1.nodes[i];

		out.nodes[i].interpolate(n1, alpha);
	}
	//    out.calculateFromTree();
}

void AnimationFrame::initTree()
{
	//nodes.resize(nodeCount);
	reset();
}

void AnimationFrame::reset(){
	for (auto &n : nodes){
		n.reset();
	}
}
