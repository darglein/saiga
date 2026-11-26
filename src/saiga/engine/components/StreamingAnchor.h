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
// A streaming centre without a Player (docs/engine_design.md §12.3; S24, G4): in a world with WorldConfig::streaming
// the chunks around the entity's root stay resident and the root's subtree is simulated (a player-less vehicle, an
// autopilot, a camera rig). A tag (saved, Replicate::Never: on a client only a client-local entity's anchor counts).
struct StreamingAnchor
{
    static constexpr const char* name = "StreamingAnchor";
    static constexpr ComponentId id   = ComponentId(EngineComponent::StreamingAnchor);
    static constexpr uint32_t version = 1;

    template <class V>
    void visit(V&)
    {
    }
};

}  // namespace Engine
}  // namespace Saiga
