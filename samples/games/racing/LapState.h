/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/racing/RacingIds.h"
#include "games/racing/Track.h"

namespace Saiga
{
namespace Game
{
/**
 * Lap timing (a component, R-STATE-1): the car has to pass the checkpoints in order (crossing a gate within its width);
 * passing checkpoint 0 after all others completes a lap.
 */
struct LapState
{
    static constexpr const char* name       = "LapState";
    static constexpr Engine::ComponentId id = Engine::ComponentId(RacingComponent::LapState);
    static constexpr uint32_t version       = 1;

    void reset(const Track& track);
    // Call every step with the car position before and after the step
    void update(const Track& track, const Vec3& before, const Vec3& after, double dt);
    // nextCheckpoint in [0, checkpoints) (a value set from outside, e.g. set_component, wraps around)
    static int wrapCheckpoint(int index, const Track& track);

    int laps           = 0;
    int nextCheckpoint = 1;
    double lapTime     = 0;
    double lastLapTime = 0;
    double bestLapTime = 0;
    int checkpointsHit = 0;

    template <class V>
    void visit(V& v)
    {
        v("laps", laps);
        v("nextCheckpoint", nextCheckpoint);
        v("lapTime", lapTime);
        v("lastLapTime", lastLapTime);
        v("bestLapTime", bestLapTime);
        v("checkpointsHit", checkpointsHit);
    }
};

}  // namespace Game
}  // namespace Saiga
