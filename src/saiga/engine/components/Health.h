/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Engine
{
// Hit points (docs/engine_design.md §9.3). Damage goes through DamageEvent and the target's onDamage; the engine does
// not destroy entities at 0 hp (game policy).
struct Health
{
    static constexpr const char* name = "Health";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Health);
    static constexpr uint32_t version = 1;

    double hp    = 100;
    double maxHp = 100;

    template <class V>
    void visit(V& v)
    {
        v("hp", hp);
        v("maxHp", maxHp);
    }
};

}  // namespace Engine
}  // namespace Saiga
