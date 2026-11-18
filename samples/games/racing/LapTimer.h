/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/racing/Track.h"

namespace Saiga
{
namespace Game
{
/**
 * Lap timing: the car has to pass the checkpoints in order (crossing a gate within its width); passing checkpoint 0
 * after all others completes a lap.
 */
class LapTimer
{
   public:
    void reset(const Track& track);
    // Call every step with the car position before and after the step
    void update(const Track& track, const Vec3& before, const Vec3& after, double dt);

    int laps           = 0;
    int nextCheckpoint = 1;
    double lapTime     = 0;
    double lastLapTime = 0;
    double bestLapTime = 0;
    int checkpointsHit = 0;
};

}  // namespace Game
}  // namespace Saiga
