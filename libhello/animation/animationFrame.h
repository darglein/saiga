#pragma once

#include <libhello/util/glm.h>
#include <vector>



class AnimationNode{
public:
    std::string name;
    std::vector<AnimationNode> children;
    mat4 testMat;

    int boneIndex = -1;

    mat4 matrix;
    mat4 transformedMatrix;

    bool keyFramed = false; //not all nodes are keyframed
    vec3 position;
    quat rotation;
    vec3 scaling;

    void reset();
    void traverse(mat4 t, std::vector<mat4> &out_boneMatrices);
};

class AnimationFrame
{
public:
    AnimationNode rootNode;

    int bones;
    std::vector<mat4> boneOffsets;
    std::vector<mat4> boneMatrices;

//    std::vector<glm::quat> boneRotations;
//    std::vector<vec3> bonePositions;
//    std::vector<vec3> boneScalings;

    std::vector<glm::dquat> boneRotations;
    std::vector<glm::dvec3> bonePositions;
    std::vector<glm::dvec3> boneScalings;

    void setBoneDeformation(std::vector<mat4> &boneMatrices);

    void test(mat4 mat);



    static void interpolate(AnimationFrame &k0, AnimationFrame &k1, float alpha, std::vector<mat4> &out_boneMatrices);
    void testd(glm::dmat4 mat);

    void calculateFromTree();
};


