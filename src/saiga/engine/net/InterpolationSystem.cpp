/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// InterpolationBuffer and interpolationSystem (phase 8, §13.4; S16). Reads: the interpolation buffers of the
// replication, the host clock, Parent and LocalTransform of remote children. Writes: RenderPose (and shifts its flags).

#include "saiga/engine/net/InterpolationSystem.h"

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

#include <algorithm>

namespace Saiga
{
namespace Engine
{
void InterpolationBuffer::push(double step, SpaceId space, const Vec3& position, const Quat& rotation, uint8_t counter)
{
    auto it =
        std::lower_bound(samples.begin(), samples.end(), step, [](const Sample& s, double t) { return s.step < t; });
    if (it != samples.end() && it->step == step)
        *it = Sample{step, space, position, rotation, counter};
    else
        samples.insert(it, Sample{step, space, position, rotation, counter});
    while (samples.size() > kMaxSamples) samples.pop_front();
}

bool InterpolationBuffer::sample(double t, SpaceId& space, Vec3& position, Quat& rotation, bool& snapped)
{
    snapped = false;
    if (samples.empty()) return false;
    // The last sample at or before t (the first one if t is earlier: nothing older is known)
    auto next =
        std::upper_bound(samples.begin(), samples.end(), t, [](double x, const Sample& s) { return x < s.step; });
    const Sample& a = next == samples.begin() ? *next : *(next - 1);
    uint8_t counter = a.counter;
    space           = a.space;
    if (next == samples.end() || next == samples.begin() || next->counter != a.counter || next->space != a.space)
    {
        // Dry (hold the newest), before the oldest, or across a teleport or a space change (no interpolation over the
        // jump: the newer sample is used from its step on)
        position = a.position;
        rotation = a.rotation;
    }
    else
    {
        const Sample& b = *next;
        const double f  = (t - a.step) / (b.step - a.step);
        position        = a.position + (b.position - a.position) * f;
        rotation        = a.rotation.slerp(f, b.rotation);
    }
    snapped        = haveSampled && (counter != sampledCounter || space != sampledSpace);
    haveSampled    = true;
    sampledCounter = counter;
    sampledSpace   = space;
    // Older samples than the one in use are not needed any more (one is kept before t)
    while (samples.size() > 2 && samples[1].step <= t) samples.pop_front();
    return true;
}

namespace
{
bool differs(const Quat& a, const Quat& b)
{
    return a.w() != b.w() || a.x() != b.x() || a.y() != b.y() || a.z() != b.z();
}
}  // namespace

void interpolationSystem(World& w, double)
{
    Registry& reg = w.registry();
    if (!reg.registered(RenderPose::id)) return;
    // The flags of the last step move on
    reg.view<RenderPose>().each(
        [](Entity, RenderPose& p)
        { p.flags = (p.flags & RenderPose::kChanged) ? RenderPose::kChangedLastStep : uint8_t(0); });
    Replication& r = w.replication();
    if (r.buffers.empty()) return;
    const double now = w.isHost() ? double(w.stepIndex()) : r.hostClock.estimate(w.stepIndex());
    const double t   = now - double(w.worldConfig().interpolationDelaySteps);
    for (auto it = r.buffers.begin(); it != r.buffers.end();)
    {
        const Entity e = reg.aliveAt(it->first);
        if (!e.valid() || e.generation != it->second.first)
        {
            it = r.buffers.erase(it);
            continue;
        }
        RenderPose* pose = reg.tryGet<RenderPose>(e);
        Vec3 p;
        Quat q;
        SpaceId space = 0;
        bool snapped  = false;
        if (pose && it->second.second.sample(t, space, p, q, snapped))
        {
            // A space change without a sampled one (the first sample after a reset) snaps too
            snapped = snapped || space != pose->space;
            if (p != pose->position || differs(q, pose->rotation) || snapped) pose->flags |= RenderPose::kChanged;
            if (snapped) pose->flags |= RenderPose::kTeleported;
            pose->position = p;
            pose->rotation = q;
            pose->space    = space;
        }
        ++it;
    }
    // Remote children: from their parent's render pose and their LocalTransform (parents first)
    for (Entity child : w.hierarchyOrder)
    {
        if (!reg.alive(child)) continue;
        RenderPose* pose     = reg.tryGet<RenderPose>(child);
        const Parent* parent = reg.tryRead<Parent>(child);
        if (!pose || !parent || !reg.alive(parent->entity)) continue;
        const RenderPose* pp = reg.tryRead<RenderPose>(parent->entity);
        if (!pp) continue;
        LocalTransform identity;
        const LocalTransform* local = reg.tryRead<LocalTransform>(child);
        if (!local) local = &identity;
        const Vec3 scale = reg.read<Transform>(parent->entity).scale();
        const Vec3 p     = pp->position + pp->rotation * scale.cwiseProduct(local->position);
        const Quat q     = pp->rotation * local->rotation;
        if (p != pose->position || differs(q, pose->rotation) || pose->space != pp->space)
            pose->flags |= RenderPose::kChanged;
        pose->flags |= pp->flags & RenderPose::kTeleported;
        if (pose->space != pp->space) pose->flags |= RenderPose::kTeleported;
        pose->position = p;
        pose->rotation = q;
        pose->space    = pp->space;
    }
}

}  // namespace Engine
}  // namespace Saiga
