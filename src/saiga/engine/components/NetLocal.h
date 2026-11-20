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
// A host entity that is never replicated (docs/engine_design.md §7.2, §13.4; S16): host-side helpers, cameras, editor
// objects. Its descendants are local too. Added before the entity's first phase-8 replication; adding or removing it
// after the spawn was sent asserts. A tag (Replicate::Never, saved).
struct NetLocal
{
    static constexpr const char* name = "NetLocal";
    static constexpr ComponentId id   = ComponentId(EngineComponent::NetLocal);
    static constexpr uint32_t version = 1;

    template <class V>
    void visit(V&)
    {
    }
};

}  // namespace Engine
}  // namespace Saiga
