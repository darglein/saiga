/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/Align.h"

#include "animation_keyframe.h"

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


class SAIGA_CORE_API Animation
{
   public:
    std::string name;

    std::vector<AnimationKeyframe> keyFrames;

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

    const AnimationKeyframe& getKeyFrame(int frameIndex);

    /**
     * Returns the interpolated frame at animation time.
     * The result will be based on the duration and the time stamps of the individual key frames.
     * The input time will be clamped to [0,duration]
     */

    void getFrame(animationtime_t time, AnimationKeyframe& out);

    /**
     * Returns the interpolated frame similar to @getFrame(float time, AnimationFrame &out);
     * The only difference is that the speed is independed of the duration.
     * A interpolation from 0 to 1 will always play the complete animation.
     */

    void getFrameNormalized(double time, AnimationKeyframe& out);

    /**
     * Prints all important information of this animation to stdout
     */
    void print();
};

class SAIGA_CORE_API AnimationSystem
{
   public:
    std::map<std::string, int> boneMap;
    std::map<std::string, int> nodeindexMap;
    AlignedVector<mat4> boneOffsets;
    AlignedVector<mat4> inverseBoneOffsets;
    std::vector<Animation> animations;


    animationtime_t animationTotalTime    = animationtime_t(0);
    animationtime_t animationTimeAtUpdate = animationtime_t(0);
    animationtime_t animationTimeAtRender = animationtime_t(0);
    int activeAnimation                   = 0;
    AnimationKeyframe currentFrame;


    float interpolate_alpha     = 0;
    int interpolating_animation = 0;
    AnimationKeyframe interpolateFrame;


    float interpolate_speed = 5;
    float animation_speed   = 1;

    void SetAnimation(int id, bool interpolate);
    void update(float dt);
    void interpolate(float dt, float alpha);
    AlignedVector<mat4> Matrices();

    void imgui();
};

}  // namespace Saiga
