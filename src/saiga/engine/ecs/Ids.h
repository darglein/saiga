/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cstddef>
#include <cstdint>

// ID types and registration modes of the ECS (docs/engine_design.md §7.2). IDs come from an explicit enum per module:
// engine components 0..63, game components 64..255.

namespace Saiga
{
namespace Engine
{
using ComponentId = uint8_t;
using EventId     = uint16_t;
using ClassId     = uint16_t;

constexpr size_t kMaxComponents           = 256;
constexpr ComponentId kFirstGameComponent = 64;

// Network replication of a component or event (§13.4; stored from S07, evaluated from S16)
enum class Replicate : uint8_t
{
    Never,
    Host,
    HostReliable,
    OwnerAuthoritative,
    Cosmetic,  // events only
};

// Saved::No: derived components that are rebuilt after loading (e.g. RenderPose)
enum class Saved : uint8_t
{
    Yes,
    No,
};

// Where a system runs (§8.2; used from S08, evaluated from S16)
enum class RunOn : uint8_t
{
    Host,
    Client,
    Both,
};

}  // namespace Engine
}  // namespace Saiga
