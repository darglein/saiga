#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <saiga/util/time.h>
#include <vector>
#include <map>

class Animation;

class SAIGA_GLOBAL AnimationNode{
public:
    //these are the local transformations of this node relative to the parent node
    //to get the absolute transformation of a single node the tree has to be traversed
    vec4 position;
    quat rotation;
    vec4 scaling;
    mat4 matrix;

    std::string name;
	std::vector<int> children;

    int index = 0;
    int boneIndex = -1;


    bool keyFramed = false; //not all nodes are keyframed


    AnimationNode(){}

    //linear interpolation of n0 and n1.
    AnimationNode(const AnimationNode& n0,const AnimationNode& n1, float alpha);

    void reset();
	void traverse(mat4 t, std::vector<mat4> &out_boneMatrices, std::vector<AnimationNode> &nodes);

};

class SAIGA_GLOBAL AnimationFrame
{
private:
    std::vector<mat4> boneMatrices;
public:
    tickd_t time = tickd_t(0); //animation time in seconds of this frame
    int nodeCount = 0; //Note: should be equal or larger than boneCount, because every bone has to be covered by a node, but a nodes can be parents of other nodes without directly having a bone.
    std::vector<AnimationNode> nodes;

    AnimationFrame(){}

    //linear interpolation of k0 and k1.
    AnimationFrame(const AnimationFrame &k0, const AnimationFrame &k1, float alpha);

    void calculateBoneMatrices(const Animation& parent);
    const std::vector<mat4>& getBoneMatrices(const Animation &parent);
    void setBoneMatrices(const std::vector<mat4> &value);
};


