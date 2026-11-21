/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/Visit.h"

#include <cstring>

namespace Saiga
{
namespace Engine
{
class Registry;
template <class T>
class ComponentPool;

/**
 * The pose of an entity in its space (docs/engine_design.md §7.4). Every entity has one (component ID 0, stored by
 * entity index). Written only through the setters (R-STATE-3): a setter changes the value and the change flag only
 * if the new value differs bitwise, so idle entities written every step cause no work downstream. The flags are not
 * visited (not saved, not hashed); the world shifts them per step (S08).
 */
class Transform
{
   public:
    static constexpr const char* name = "Transform";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Transform);
    static constexpr uint32_t version = 1;

    Transform()                     = default;
    Transform(const Transform&)     = default;
    Transform(Transform&&) noexcept = default;

    SpaceId space() const { return sp; }
    const Vec3& position() const { return p; }
    const Quat& rotation() const { return r; }
    const Vec3& scale() const { return s; }

    void setPosition(const Vec3& v)
    {
        if (differs(p, v))
        {
            p               = v;
            changedThisStep = 1;
        }
    }
    void setRotation(const Quat& v)
    {
        if (differs(r, v))
        {
            r               = v;
            changedThisStep = 1;
        }
    }
    void setScale(const Vec3& v)
    {
        if (differs(s, v))
        {
            s               = v;
            changedThisStep = 1;
        }
    }
    // A jump: no interpolation from the previous pose
    void teleport(const Vec3& position, const Quat& rotation)
    {
        p               = position;
        r               = rotation;
        changedThisStep = 1;
        teleported      = 1;
        teleports++;
    }

    // Change flags (engine-internal readers: hierarchy, grid, extract; S08)
    bool changedInThisStep() const { return changedThisStep != 0; }
    bool changedInLastStep() const { return changedLastStep != 0; }
    bool wasTeleported() const { return teleported != 0; }
    // Teleported in the previous step: readers that run before a late teleport (extract in phase 8 of the last step)
    // still see it, and phase 6 passes it on to children that did not follow yet
    bool wasTeleportedLastStep() const { return teleportedLastStep != 0; }
    // Counts teleports (wrapping; not saved or hashed): a reader that handled one teleport tells it from a later one in
    // the same step (the extract's catch-up, S19b)
    uint8_t teleportCount() const { return teleports; }

    // For detached copies only (saves, hashes, network decoding into a temporary): reading through visit writes the
    // fields without the change flags. Applying a decoded pose to a live entity goes through the setters or teleport
    // (S08/S16).
    template <class V>
    void visit(V& v)
    {
        v("space", sp);
        // Replication sends the pose quantized (§13.4); saves and the hash keep the exact values
        v("position", p, Quantize{1e-4});
        v("rotation", r, Quantize{1e-5});
        v("scale", s);
    }

   private:
    friend class Registry;
    // The World's space primitive (setSubtreeSpace, spawns, state entities; S21) is the only writer of the space
    friend class World;
    // MSVC instantiates ComponentPool<Transform> (its swap-and-pop assignment) although Transform has no pool
    template <class T>
    friend class ComponentPool;
    // Whole-object assignment would bypass the setters and the change flags (R-STATE-3); only the registry resets a
    // reused slot. Copy and move construction stay public (storage, tests).
    Transform& operator=(const Transform&)     = default;
    Transform& operator=(Transform&&) noexcept = default;

    // A space change of the World (S21): the space and the change flag, no teleport flag (the root's teleport sets it)
    void setSpaceInternal(SpaceId space)
    {
        if (sp == space) return;
        sp              = space;
        changedThisStep = 1;
    }

    // Bitwise per double: -0.0 vs 0.0 and NaN payloads count as changes, equal bits never do (no padding compared)
    static bool differs(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) != 0; }
    static bool differs(const Vec3& a, const Vec3& b)
    {
        return differs(a(0), b(0)) || differs(a(1), b(1)) || differs(a(2), b(2));
    }
    static bool differs(const Quat& a, const Quat& b)
    {
        return differs(a.w(), b.w()) || differs(a.x(), b.x()) || differs(a.y(), b.y()) || differs(a.z(), b.z());
    }

    SpaceId sp              = 0;
    Vec3 p                  = Vec3::Zero();
    Quat r                  = Quat::Identity();
    Vec3 s                  = Vec3::Ones();
    uint8_t changedThisStep = 0, changedLastStep = 0, teleported = 0, teleportedLastStep = 0;
    uint8_t teleports = 0;
};

}  // namespace Engine
}  // namespace Saiga
