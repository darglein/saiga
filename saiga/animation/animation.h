#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <saiga/animation/animationFrame.h>


class SAIGA_GLOBAL Animation
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

    void getFrame(float f, AnimationFrame &out);

    void update();
    void setKeyFrame(int i);
    void setKeyFrame(float f);
};


