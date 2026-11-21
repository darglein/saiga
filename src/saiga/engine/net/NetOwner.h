/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// The owning peer of a replicated root (docs/engine_design.md §13.3; S16): set by World::setOwner on the host before
// the entity's spawn is sent, never changed afterwards; descendants inherit their root's owner (NetOwner on a child
// asserts). Entities without it belong to the host (peer 0). Read-only through the World (setOwner).
struct NetOwner
{
    static constexpr const char* name = "NetOwner";
    static constexpr ComponentId id   = ComponentId(EngineComponent::NetOwner);
    static constexpr uint32_t version = 1;

    uint8_t peer = 0;

    template <class V>
    void visit(V& v)
    {
        v("peer", peer);
    }
};

}  // namespace Engine
}  // namespace Saiga
