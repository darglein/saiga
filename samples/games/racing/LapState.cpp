/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Lap timing of the racing tech demo: checkpoints in order, laps, lap times.

#include "games/racing/LapState.h"

#include <cstdint>

namespace Saiga
{
namespace Game
{
void LapState::reset(const Track& track)
{
    *this          = LapState();
    nextCheckpoint = track.checkpoints().size() > 1 ? 1 : 0;
}

int LapState::wrapCheckpoint(int index, const Track& track)
{
    const int64_t n = int64_t(track.checkpoints().size());
    if (n == 0) return 0;
    return int(((int64_t(index) % n) + n) % n);
}

void LapState::update(const Track& track, const Vec3& before, const Vec3& after, double dt)
{
    lapTime += dt;
    const auto& gates = track.checkpoints();
    if (gates.empty()) return;
    nextCheckpoint = wrapCheckpoint(nextCheckpoint, track);
    const auto& g  = gates[nextCheckpoint];
    // Crossing the gate's line (forward through it) within the track width
    Vec3 a(before(0) - g.center(0), 0, before(2) - g.center(2));
    Vec3 b(after(0) - g.center(0), 0, after(2) - g.center(2));
    double da = a.dot(g.tangent), db = b.dot(g.tangent);
    if (!(da < 0 && db >= 0)) return;
    // The width check applies where the step crosses the line (not at its end point)
    double t = da / (da - db);
    Vec3 x   = a + (b - a) * t;
    if ((x - g.tangent * x.dot(g.tangent)).norm() > track.width * 0.5 + 3) return;
    checkpointsHit++;
    if (nextCheckpoint == 0)
    {
        laps++;
        lastLapTime = lapTime;
        bestLapTime = bestLapTime > 0 ? std::min(bestLapTime, lapTime) : lapTime;
        lapTime     = 0;
    }
    nextCheckpoint = (nextCheckpoint + 1) % int(gates.size());
}

}  // namespace Game
}  // namespace Saiga
