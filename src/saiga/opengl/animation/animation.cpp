/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "animation.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <iostream>
namespace Saiga
{
const AnimationFrame& Animation::getKeyFrame(int frameIndex)
{
    SAIGA_ASSERT(frameIndex >= 0 && frameIndex < frameCount);
    return keyFrames[frameIndex];
}

void Animation::getFrame(animationtime_t time, AnimationFrame& out)
{
    // here time is given in animation time base
    time = std::max(std::min(time, duration), animationtime_t(0));

    // seach for correct frame interval
    int frame = 0;
    for (AnimationFrame& af : keyFrames)
    {
        if (af.time >= time)
        {
            break;
        }
        frame++;
    }
    int prevFrame = std::max(0, frame - 1);

    AnimationFrame& k0 = keyFrames[prevFrame];
    AnimationFrame& k1 = keyFrames[frame];

    if (frame == prevFrame)
    {
        out = k0;
        return;
    }

    float alpha = ((time - k0.time).count() / (k1.time - k0.time).count());
    out         = AnimationFrame(k0, k1, alpha);
}

void Animation::getFrameNormalized(double time, AnimationFrame& out)
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
    for (AnimationFrame& af : keyFrames)
    {
        std::cout << af.time.count() << ", ";
    }
    std::cout << "]" << std::endl;
}

}  // namespace Saiga
