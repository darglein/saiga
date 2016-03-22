#include "saiga/animation/animationFrame.h"

#include <glm/gtx/quaternion.hpp>
#include <iostream>

using std::cout;
using std::endl;

void AnimationFrame::calculateFromTree()
{
    rootNode.traverse(mat4(),boneMatrices);
    for(unsigned int i=0;i<boneMatrices.size();++i){
        boneMatrices[i] =  boneMatrices[i] * boneOffsets[i];
    }
}

void AnimationFrame::interpolate(AnimationFrame &k0, AnimationFrame &k1, AnimationFrame& out, float alpha)
{
    out = k0;
    out.initTree();

    for(int i=0;i<k0.nodeCount;++i){
        AnimationNode* n1 = k1.nodes[i];

        out.nodes[i]->interpolate(*n1,alpha);
    }




//    out.calculateFromTree();

}

void AnimationFrame::initTree()
{
    nodes.resize(nodeCount);
    rootNode.initTree(nodes);

}

void AnimationNode::interpolate(AnimationNode &other, float alpha)
{
    if(this->keyFramed){
        rotation = glm::slerp(rotation,other.rotation,alpha);
        rotation = glm::normalize(rotation);
        scaling = glm::mix(scaling,other.scaling,alpha);
        position = glm::mix(position,other.position,alpha);
    }
}

void AnimationNode::initTree(std::vector<AnimationNode *> &nodes)
{
    nodes[this->index] = this;
    for(AnimationNode &an : children){
        an.initTree(nodes);
    }
}

void AnimationNode::reset()
{
    position = vec3(0);
    rotation = quat();
    scaling = vec3(1);
    transformedMatrix = mat4();
    for(AnimationNode &an : children){
        an.reset();
    }
}



void AnimationNode::traverse(mat4 m,  std::vector<mat4> &out_boneMatrices)
{
    //    cout<<"AnimationNode::traverse(mat4 m,  std::vector<mat4> out_boneMatrices)"<<endl;

    if(keyFramed){
        glm::mat4 t = glm::translate(glm::mat4(),position);
        glm::mat4 r = glm::mat4_cast(rotation);
        glm::mat4 s = glm::scale(glm::mat4(),scaling);
        matrix = t*s*r;
    }


    transformedMatrix = m*matrix;

    if(boneIndex!=-1){
        out_boneMatrices[boneIndex] = transformedMatrix;
    }

    for(AnimationNode &an : children){
        an.traverse(transformedMatrix,out_boneMatrices);
    }
}
