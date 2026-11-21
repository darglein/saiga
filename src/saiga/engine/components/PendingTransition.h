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
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// A requested transition of a root to another space (docs/engine_design.md §12.4; S21), added by
// World::requestTransition and completed by spaceTransitionSystem (host, phase 1 of the next step; from S24 when the
// arrival is complete). The target and the pose are copied from the SpaceLink at the request (a link edited or
// destroyed later changes nothing). Replicate::Never, saved: a save between the request and the next step keeps it
// (R-STATE-1).
struct PendingTransition
{
    static constexpr const char* name = "PendingTransition";
    static constexpr ComponentId id   = ComponentId(EngineComponent::PendingTransition);
    static constexpr uint32_t version = 1;

    Entity link;  // for the event; may be dead when the transition completes
    SpaceId target = 0;
    Vec3 position  = Vec3::Zero();
    Quat rotation  = Quat::Identity();

    template <class V>
    void visit(V& v)
    {
        v("link", link);
        v("target", target);
        v("position", position);
        v("rotation", rotation);
    }
};

}  // namespace Engine
}  // namespace Saiga
