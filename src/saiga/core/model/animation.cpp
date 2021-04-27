/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "animation.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"

#include <algorithm>
#include <iostream>
namespace Saiga
{
const AnimationKeyframe& Animation::getKeyFrame(int frameIndex)
{
    SAIGA_ASSERT(frameIndex >= 0 && frameIndex < frameCount);
    return keyFrames[frameIndex];
}

void Animation::getFrame(animationtime_t time, AnimationKeyframe& out)
{
    // here time is given in animation time base
    time = std::max(std::min(time, duration), animationtime_t(0));

    // seach for correct frame interval
    int frame = 0;
    for (AnimationKeyframe& af : keyFrames)
    {
        if (af.time >= time)
        {
            break;
        }
        frame++;
    }
    int prevFrame = std::max(0, frame - 1);

    AnimationKeyframe& k0 = keyFrames[prevFrame];
    AnimationKeyframe& k1 = keyFrames[frame];

    if (frame == prevFrame)
    {
        out = k0;
        return;
    }

    float alpha = ((time - k0.time).count() / (k1.time - k0.time).count());
    out         = AnimationKeyframe(k0, k1, alpha);
}

void Animation::getFrameNormalized(double time, AnimationKeyframe& out)
{
    time = clamp(time, 0.0, 1.0);
    SAIGA_ASSERT(time >= 0 && time <= 1);
    getFrame(duration * time, out);
}

void Animation::print()
{
    std::cout << "[Animation] " + name << " Frames=" << frameCount << " duration=" << duration.count() << "s"
              << std::endl;
    std::cout << "\tKeyframes: [";
    for (AnimationKeyframe& af : keyFrames)
    {
        std::cout << af.time.count() << ", ";
    }
    std::cout << "]" << std::endl;
}

void AnimationSystem::SetAnimation(int id, bool interpolate)
{
    SAIGA_ASSERT(id >= 0 && id < (int)animations.size());

    if (interpolate && id != activeAnimation)
    {
        interpolating_animation = activeAnimation;
        interpolateFrame        = currentFrame;
        interpolate_alpha       = 1;
    }
    activeAnimation    = id;
    animationTotalTime = animations[id].duration;
}

void AnimationSystem::update(float dt)
{
    animationTimeAtUpdate += animationtime_t(dt * animation_speed);
    interpolate_alpha -= dt * interpolate_speed;
    interpolate_alpha = std::max<float>(interpolate_alpha, 0);
    // loop animation constantly
    if (animationTimeAtUpdate >= animationTotalTime) animationTimeAtUpdate -= animationTotalTime;
}

void AnimationSystem::interpolate(float dt, float alpha)
{
    animationTimeAtRender = animationTimeAtUpdate + animationtime_t(animation_speed * dt * alpha);
    if (animationTimeAtRender >= animationTotalTime) animationTimeAtRender -= animationTotalTime;

    animations[activeAnimation].getFrame(animationTimeAtRender, currentFrame);

    if (interpolate_alpha > 0)
    {
        currentFrame = AnimationKeyframe(currentFrame, interpolateFrame, interpolate_alpha);
    }
}

AlignedVector<mat4> AnimationSystem::Matrices()
{
    return currentFrame.getBoneMatrices(animations[activeAnimation]);
}

void AnimationSystem::imgui()
{
    ImGui::Text("AnimationSystem");
    ImGui::Text("Active Animation %d", activeAnimation);
    ImGui::Text("Animation Time %f", animationTimeAtUpdate.count());
    ImGui::Text("Animation Interpolate %f", interpolate_alpha);
    ImGui::InputFloat("interpolate_speed", &interpolate_speed);
    ImGui::InputFloat("animation_speed", &animation_speed);
}

}  // namespace Saiga
