/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstdint>

// The IDs of the engine's components (docs/engine_design.md §7.2): 0..63, the enum grows per stage; game components
// start at 64 (kFirstGameComponent).

namespace Saiga
{
namespace Engine
{
enum class EngineComponent : uint8_t
{
    Transform = 0,
    Parent,
    LocalTransform,
    Behaviour,
    Persistent,
    Player,
    Health,
    PlayerInput,
    CharacterIntent,
    NetOwner,           // S16
    NetLocal,           // S16
    RenderPose,         // S16
    RenderInstance,     // S19b
    SpaceLink,          // S21
    PendingTransition,  // S21
    WorldTime,          // S36
    Weather,            // S36
    Count
};

// The engine's game object classes (ClassId 0..63; none yet); game classes start at 64 (kFirstGameClass)
enum class EngineClass : uint16_t
{
    Count
};

}  // namespace Engine
}  // namespace Saiga
