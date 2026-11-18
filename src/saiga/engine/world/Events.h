/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstdint>

// The engine's events (docs/engine_design.md §7.7): IDs 0..63; the enum grows with the stages (DamageEvent,
// InteractEvent, ContactEvent from S10/S28). Game events start at 64.

namespace Saiga
{
namespace Engine
{
class EventRegistry;

enum class EngineEvent : uint16_t
{
    Count
};

// Registers the engine's events (none yet)
inline void registerEngineEvents(EventRegistry&) {}

}  // namespace Engine
}  // namespace Saiga
