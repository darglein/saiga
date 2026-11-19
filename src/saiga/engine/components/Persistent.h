/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/PersistentId.h"

namespace Saiga
{
namespace Engine
{
// A referable identity that survives saving and (from S25) unloading (docs/engine_design.md §9.6). Assigned by
// world.makePersistent; world.resolve(id) finds the entity. Not added or changed directly (the World keeps the index).
struct Persistent
{
    static constexpr const char* name = "Persistent";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Persistent);
    static constexpr uint32_t version = 1;

    PersistentId persistentId;

    template <class V>
    void visit(V& v)
    {
        v("persistentId", persistentId);
    }
};

}  // namespace Engine
}  // namespace Saiga
