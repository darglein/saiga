/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <algorithm>

namespace Saiga
{
namespace Game
{
/**
 * A fixed-step clock (double dt): advance() runs the step function for every full dt in the accumulated frame time
 * (at most maxSteps per call, the rest is dropped), alpha() is the interpolation factor between the last two states.
 * The samples run their games in MainLoop::update (fixed 60 Hz); tests and headless runs use this clock.
 * Transitional: World::advance (engine_design.md §8.3) replaces it in S08/S13.
 */
struct FixedStep
{
    double dt          = 1.0 / 60.0;
    double accumulator = 0;
    int maxSteps       = 8;

    template <typename F>
    int advance(double frameDt, F&& step)
    {
        accumulator += std::max(frameDt, 0.0);
        int steps = 0;
        while (accumulator >= dt && steps < maxSteps)
        {
            step(dt);
            accumulator -= dt;
            steps++;
        }
        if (steps == maxSteps) accumulator = std::min(accumulator, dt);
        return steps;
    }
    double alpha() const { return std::min(accumulator / dt, 1.0); }
};

}  // namespace Game
}  // namespace Saiga
