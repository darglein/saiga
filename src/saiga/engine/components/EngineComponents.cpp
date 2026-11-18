/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's component registration list.

#include "saiga/engine/components/EngineComponents.h"

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/ecs/Registry.h"

namespace Saiga
{
namespace Engine
{
void registerEngineComponents(Registry& registry)
{
    registry.registerComponent<Parent>(Replicate::Host);
    registry.registerComponent<LocalTransform>(Replicate::Host);
}

}  // namespace Engine
}  // namespace Saiga
