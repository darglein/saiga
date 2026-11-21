/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class World;

// Phase 1 (host; engine-registered, the first system of registerEngineSystems; S21, §12.4): completes every
// PendingTransition (view order) with moveToSpace to its target at its pose, emits SpaceTransitionEvent and removes the
// component (a command). A pending transition that is no longer valid (the entity became a child or a state entity,
// the target is not a space: only from a tool's set_component, the load checks saves) is removed with a log line.
// Before S24 every request completes in phase 1 of the step after the request.
SAIGA_ENGINE_API void spaceTransitionSystem(World& world, double dt);

}  // namespace Engine
}  // namespace Saiga
