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

    //duration of animation in seconds
    float duration = 1;

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
     * The result will be based on the duration and the time stamps of the individual key frames.
     * The input time will be clamped to [0,duration]
     */

//    void getFrame(float time, AnimationFrame &out);
    void getFrame2(float time, AnimationFrame &out);

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


    /**
     * Prints all important information of this animation to stdout
     */
    void print();

};


