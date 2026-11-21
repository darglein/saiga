/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// A client's estimate of the host step (docs/engine_design.md §13.4; S16): hostStep = localStep + offset, from the host
// steps of snapshots and the Ping/Pong round trip, both in local steps (never wall clock). The offset slews by at most
// kMaxSlew steps per step towards the newest estimate and jumps only when the error exceeds kResyncSteps (or for the
// first estimate).

#pragma once

#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API HostClock
{
   public:
    static constexpr double kMaxSlew     = 1.0 / 8.0;
    static constexpr double kResyncSteps = 30;
    static constexpr int kPingInterval   = 30;

    // A snapshot of hostStep arrived in local step localStep
    void observeSnapshot(uint64_t hostStep, uint64_t localStep);
    // A Pong to the Ping sent in local step pingStep arrived in local step localStep
    void observePong(uint64_t pingStep, uint64_t hostStep, uint64_t localStep);
    // Once per local step: the offset moves towards the target (at most kMaxSlew)
    void update();
    void reset() { *this = HostClock(); }

    bool valid() const { return have; }
    double offset() const { return current; }
    double target() const { return goal; }
    double roundTrip() const { return rtt; }
    double estimate(uint64_t localStep) const { return double(localStep) + current; }

   private:
    void observe(double newTarget);
    bool have      = false;
    double current = 0, goal = 0, rtt = 0;
};

}  // namespace Engine
}  // namespace Saiga
