/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/opengl/animation/animationFrame.h"
#include "saiga/core/time/time.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
/**
 * Animation time:
 * The duration of an animation is the time between the first and the last keyframe.
 * The keyframes between the first and the last can shown at arbitrary time stamps.
 *
 * For example a 10 second animation could have the following 4 keyframes:
 * [0,1,6,10]
 *
 * Interpolation between the keyframes is always linear in the corresponding interval.
 * Time '7' in the above example would interpolate frame 3 and frame 4 with alpha=0.25.
 * For doing this interpolation use getFrame(..)
 */


class SAIGA_OPENGL_API Animation
{
   public:
    std::string name;

    std::vector<AnimationFrame> keyFrames;

    // number of keyframes
    int frameCount = 0;

    // speed at which this animation should be played. Unused in this class.
    float animationSpeed = 1.0f;

    // duration of animation
    animationtime_t duration = animationtime_t(1);


    int boneCount = 0;
    AlignedVector<mat4> boneOffsets;

    /**
     * Returns the keyframe at the given index.
     */

    const AnimationFrame& getKeyFrame(int frameIndex);

    /**
     * Returns the interpolated frame at animation time.
     * The result will be based on the duration and the time stamps of the individual key frames.
     * The input time will be clamped to [0,duration]
     */

    void getFrame(animationtime_t time, AnimationFrame& out);

    /**
     * Returns the interpolated frame similar to @getFrame(float time, AnimationFrame &out);
     * The only difference is that the speed is independed of the duration.
     * A interpolation from 0 to 1 will always play the complete animation.
     */

    void getFrameNormalized(double time, AnimationFrame& out);

    /**
     * Prints all important information of this animation to stdout
     */
    void print();
};

}  // namespace Saiga
