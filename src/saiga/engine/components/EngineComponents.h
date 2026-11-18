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

// Registers the engine's components (Transform is built into the Registry): Parent, LocalTransform. Called by the
// World before the game's list (docs/engine_design.md §18.2).
SAIGA_ENGINE_API void registerEngineComponents(Registry& registry);

}  // namespace Engine
}  // namespace Saiga
