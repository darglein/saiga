#pragma once

#include <libhello/util/glm.h>
#include <vector>


class AnimationFrame
{
public:
    int bones;
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
};


