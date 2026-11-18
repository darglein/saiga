/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

#include <cstdint>

namespace Saiga
{
namespace Engine
{
// The phases of one step (docs/engine_design.md §8.1), in execution order. Commands are applied after every phase.
enum class Phase : uint8_t
{
    None = 0,   // between steps
    Input,      // 1: local input and received messages -> components
    Events,     // 2: the previous step's events to GameObject handlers
    Behaviour,  // 3: GameObject::update
    Gameplay,   // 4: engine and game systems
    Physics,    // 5: physics step, contacts
    Transform,  // 6: hierarchy, spatial grid, chunk membership
    Streaming,  // 7: chunk results
    Extract,    // 8: render sink, replication, structural log consumers
};

constexpr int kFirstPhase = int(Phase::Input);
constexpr int kLastPhase  = int(Phase::Extract);

// Phases 2-5 see only Simulated entities in world.view (§12.3)
constexpr bool simulatedOnly(Phase p)
{
    return p >= Phase::Events && p <= Phase::Physics;
}

SAIGA_ENGINE_API const char* phaseName(Phase p);

}  // namespace Engine
}  // namespace Saiga
