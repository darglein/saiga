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

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// The pose to render an entity with when its Transform is not the local truth (docs/engine_design.md §10.3, §13.4;
// S16): written in phase 8 by the interpolation system for remote entities on clients and for client-owned entities on
// the host, from the interpolation buffer at estimated host step - interpolationDelaySteps. flags: kChanged (this
// step), kChangedLastStep, kTeleported (no interpolation from the previous pose); shifted every step by the
// interpolation system. space (version 2, S21): the space of the pose (a space change is a teleport, never
// interpolated). Replicate::Never, Saved::No (rebuilt from the buffer).
struct RenderPose
{
    static constexpr const char* name = "RenderPose";
    static constexpr ComponentId id   = ComponentId(EngineComponent::RenderPose);
    static constexpr uint32_t version = 2;

    static constexpr uint8_t kChanged         = 1;
    static constexpr uint8_t kChangedLastStep = 2;
    static constexpr uint8_t kTeleported      = 4;

    Vec3 position = Vec3::Zero();
    Quat rotation = Quat::Identity();
    uint8_t flags = 0;
    SpaceId space = 0;

    template <class V>
    void visit(V& v)
    {
        v("position", position);
        v("rotation", rotation);
        v("flags", flags);
        v("space", space);
    }
};

}  // namespace Engine
}  // namespace Saiga
