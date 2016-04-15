#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <vector>
#include <map>

class SAIGA_GLOBAL AnimationNode{
public:
    std::string name;
    //std::vector<AnimationNode*> children;
	std::vector<int> children;

    int index = 0;
    int boneIndex = -1;

    mat4 matrix;
//    mat4 transformedMatrix;

    bool keyFramed = false; //not all nodes are keyframed
    vec3 position;
    quat rotation;
    vec3 scaling;

    void interpolate(AnimationNode& other, float alpha);

	void reset();
	void traverse(mat4 t, std::vector<mat4> &out_boneMatrices, std::vector<AnimationNode> &nodes);

};

class SAIGA_GLOBAL AnimationFrame
{
public:
    int nodeCount = 0;
    std::vector<AnimationNode> nodes;
    int rootNode;

    int bones;
    std::vector<mat4> boneOffsets;
    std::vector<mat4> boneMatrices;

    AnimationFrame();
    AnimationFrame(const AnimationFrame &other);


    static void interpolate(AnimationFrame &k0, AnimationFrame &k1, AnimationFrame &out, float alpha);

    void calculateFromTree();
    void initTree();
	void reset();
};


