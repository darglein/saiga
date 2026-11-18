/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"

namespace Saiga
{
namespace Engine
{
// The parent in the Transform hierarchy (docs/engine_design.md §7.4): the entity's Transform follows the parent's
// Transform combined with its LocalTransform (phase 6). Destroying the parent destroys the child.
struct Parent
{
    static constexpr const char* name = "Parent";
    static constexpr ComponentId id   = ComponentId(EngineComponent::Parent);
    static constexpr uint32_t version = 1;

    Entity entity;

    template <class V>
    void visit(V& v)
    {
        v("entity", entity);
    }
};

}  // namespace Engine
}  // namespace Saiga
