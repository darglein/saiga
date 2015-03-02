#pragma once

#include <libhello/util/glm.h>
#include <libhello/animation/animationFrame.h>


class Animation
{
public:
    std::string name;
    AnimationFrame restPosition;


    std::vector<AnimationFrame> animationFrames;
    std::vector<mat4> boneMatrices;

    int frameCount = 0;

    int animfps = 30, animlen = 0;
    float animtick = 0;

    Animation();

    void update();
    void setKeyFrame(int i);
    void setKeyFrame(float f);
};


