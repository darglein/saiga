#pragma once

#include <saiga/config.h>
#include <saiga/util/glm.h>
#include <saiga/animation/animationFrame.h>


class SAIGA_GLOBAL Animation
{
public:
    std::string name;

    std::vector<AnimationFrame> keyFrames;

    //number of keyframes
    int frameCount = 0;

    //speed at which this animation should be played. Unused in this class.
    float animationSpeed = 1.0f;

    //false if there should be an interpolation between the last and the first node.
    //typically if the first and the last node are equal it should be true, otherwise false.
    bool skipLastFrame = false;


    /**
     * returns frameCount or frameCount-1 depending on skipLastFrame
     */

    int getActiveFrames();

    /**
     * Returns the keyframe at the given index.
     */

    const AnimationFrame& getKeyFrame(int frameIndex);

    /**
     * Returns the interpolated frame at time 'time'.
     * With time=0,1,2... the keyFrames 0,1,2... are returned.
     * A time=1.5f will interpolate between the frames 1 and 2 with an alpha of 0.5f.
     */

    void getFrame(float time, AnimationFrame &out);

    /**
     * Returns the interpolated frame similar to @getFrame(float time, AnimationFrame &out);
     * The only difference is that the speed is independed of the number of keyframes used.
     * A interpolation from 0 to 1 will always play the complete animation.
     */

    void getFrameNormalized(float time, AnimationFrame &out);

    /**
     * Returns the interpolated animation frame of 'frame0' and 'frame1' at
     * time 'alpha'.
     * Alpha should be in the range [0,1] for reasonable results.
     */

    void getFrame(int frame0, int frame1, float alpha, AnimationFrame &out);

};


