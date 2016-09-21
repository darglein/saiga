#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <vector>
#include <map>

class SAIGA_GLOBAL AnimationNode{
public:
    std::string name;
	std::vector<int> children;

    int index = 0;
    int boneIndex = -1;

    mat4 matrix;
//    mat4 transformedMatrix;

    bool keyFramed = false; //not all nodes are keyframed
    vec3 position;
    quat rotation;
    vec3 scaling;

    void interpolate(const AnimationNode& other, float alpha);

    void reset();
	void traverse(mat4 t, std::vector<mat4> &out_boneMatrices, std::vector<AnimationNode> &nodes);

};

class SAIGA_GLOBAL AnimationFrame
{
private:
    std::vector<mat4> boneMatrices;
public:
    float time = 0; //animation time in seconds of this frame

    int nodeCount = 0; //Note: should be equal or larger than boneCount, because every bone has to be covered by a node, but a nodes can be parents of other nodes without directly having a bone.
    std::vector<AnimationNode> nodes;

#define AnimationFrame_ROOT_NODE 0
//    static constexpr int rootNode = 0;

    int boneCount = 0;
    std::vector<mat4> boneOffsets;

    AnimationFrame();

    void calculateBoneMatrices();

    const std::vector<mat4>& getBoneMatrices();
    void setBoneMatrices(const std::vector<mat4> &value);
    /**
     * Linear interpolation between two AnimationFrames. Used by Animation class to create smooth keyframe based animations.
     */

    static void interpolate(const AnimationFrame &k0, const AnimationFrame &k1, AnimationFrame &out, float alpha);

};


