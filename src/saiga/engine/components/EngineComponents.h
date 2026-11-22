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
class Registry;
struct WorldSetup;

// Registers the engine's components (Transform is built into the Registry): Parent, LocalTransform, ... Called by the
// World before the game's list (docs/engine_design.md §18.2). Every engine component is always registered (the prefix
// rule of Registry::validateRegistration); the setup only chooses the kind of WorldTime and Weather (S36, D4): global /
// space state with setup.environment.enabled, else plain components.
SAIGA_ENGINE_API void registerEngineComponents(Registry& registry, const WorldSetup& setup);

}  // namespace Engine
}  // namespace Saiga
