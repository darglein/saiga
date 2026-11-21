/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// HostClock: the slewed host step estimate of a client.

#include "saiga/engine/net/HostClock.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Engine
{
void HostClock::observe(double newTarget)
{
    goal = newTarget;
    // The first estimate and a resync (an error beyond kResyncSteps) jump; otherwise update() slews
    if (!have || std::abs(goal - current) > kResyncSteps)
    {
        current = goal;
        have    = true;
    }
}

void HostClock::observeSnapshot(uint64_t hostStep, uint64_t localStep)
{
    // The snapshot left the host half a round trip ago
    observe(double(hostStep) + 0.5 * rtt - double(localStep));
}

void HostClock::observePong(uint64_t pingStep, uint64_t hostStep, uint64_t localStep)
{
    if (localStep < pingStep) return;
    const double measured = double(localStep - pingStep);
    rtt                   = rtt == 0 ? measured : 0.75 * rtt + 0.25 * measured;
    observe(double(hostStep) + 0.5 * measured - double(localStep));
}

void HostClock::update()
{
    if (!have) return;
    current += std::max(-kMaxSlew, std::min(kMaxSlew, goal - current));
}

}  // namespace Engine
}  // namespace Saiga
