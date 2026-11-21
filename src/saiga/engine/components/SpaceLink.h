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
// An entrance to another space (docs/engine_design.md §12.4; S21): World::requestTransition(e, link) moves e to target
// at the arrival pose (in the target space). Who uses it and how (key, distance, prompts) is game policy: the optional
// spaceLinkSystem turns an InteractEvent with this entity as the target into the request. Replicate::Host (clients show
// prompts), saved.
struct SpaceLink
{
    static constexpr const char* name = "SpaceLink";
    static constexpr ComponentId id   = ComponentId(EngineComponent::SpaceLink);
    static constexpr uint32_t version = 1;

    SpaceId target       = 0;
    Vec3 arrival         = Vec3::Zero();
    Quat arrivalRotation = Quat::Identity();

    template <class V>
    void visit(V& v)
    {
        v("target", target);
        v("arrival", arrival);
        v("arrivalRotation", arrivalRotation);
    }
};

}  // namespace Engine
}  // namespace Saiga
