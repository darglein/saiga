/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Types.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Engine
{
// What a character wants to do this step (docs/engine_design.md §9.3), computed on each peer (not replicated): written
// by playerControlSystem or game AI, executed by characterMoveSystem.
struct CharacterIntent
{
    static constexpr const char* name = "CharacterIntent";
    static constexpr ComponentId id   = ComponentId(EngineComponent::CharacterIntent);
    static constexpr uint32_t version = 1;

    Vec3 desiredVelocity = Vec3::Zero();
    Entity target;

    template <class V>
    void visit(V& v)
    {
        v("desiredVelocity", desiredVelocity);
        v("target", target);
    }
};

}  // namespace Engine
}  // namespace Saiga
