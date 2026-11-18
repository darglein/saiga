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

#include <cstring>

namespace Saiga
{
namespace Engine
{
class Registry;

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
    }

    // Change flags (engine-internal readers: hierarchy, grid, extract; S08)
    bool changedInThisStep() const { return changedThisStep != 0; }
    bool changedInLastStep() const { return changedLastStep != 0; }
    bool wasTeleported() const { return teleported != 0; }

    template <class V>
    void visit(V& v)
    {
        v("space", sp);
        v("position", p);
        v("rotation", r);
        v("scale", s);
    }

   private:
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
    uint8_t changedThisStep = 0, changedLastStep = 0, teleported = 0;
};

}  // namespace Engine
}  // namespace Saiga
