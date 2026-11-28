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
// The island count last reported by IslandLimitEvent (docs/engine_design.md §12.3; S24, G14): a global (world-state
// component) of worlds with WorldConfig::streaming, else a plain component no entity has. Written by the host's
// streaming system at the phase-7 flush whenever the count differs from it (together with the event), so a load
// continues the reporting exactly. Saved, Replicate::Never.
struct IslandState
{
    static constexpr const char* name = "IslandState";
    static constexpr ComponentId id   = ComponentId(EngineComponent::IslandState);
    static constexpr uint32_t version = 1;

    uint32_t reportedCount = 0;

    template <class V>
    void visit(V& v)
    {
        v("reportedCount", reportedCount);
    }
};

}  // namespace Engine
}  // namespace Saiga
