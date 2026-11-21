/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The interpolation buffer of one entity and the interpolation system (docs/engine_design.md §13.4, §10.3; S16).
// interpolationSystem (phase 8, after replication; called by the World): reads the buffers (received poses by host
// step), the HostClock and LocalTransform; writes RenderPose of remote entities on clients and of client-owned entities
// on the host (children from their parent's RenderPose and LocalTransform).

#pragma once

#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/export.h"

#include <cstdint>
#include <deque>

namespace Saiga
{
namespace Engine
{
class World;

// Received poses by host step (every snapshot or owner state, not only the newest), sampled linearly in position and
// with slerp in rotation; a change of the teleport counter between two samples snaps; past the newest sample the last
// pose is held (dry).
class SAIGA_ENGINE_API InterpolationBuffer
{
   public:
    struct Sample
    {
        double step;
        Vec3 position;
        Quat rotation;
        uint8_t counter;
    };
    static constexpr size_t kMaxSamples = 64;

    // Kept in step order; a sample at an existing step replaces it
    void push(double step, const Vec3& position, const Quat& rotation, uint8_t counter);
    // false: empty. snapped: a teleport counter change at or before t since the last sample() call
    bool sample(double t, Vec3& position, Quat& rotation, bool& snapped);
    bool empty() const { return samples.empty(); }
    size_t size() const { return samples.size(); }
    void clear() { samples.clear(); }

   private:
    std::deque<Sample> samples;
    bool haveSampled       = false;
    uint8_t sampledCounter = 0;
};

SAIGA_ENGINE_API void interpolationSystem(World& world, double dt);

}  // namespace Engine
}  // namespace Saiga
