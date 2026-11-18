/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The engine's component registration list.

#include "saiga/engine/components/EngineComponents.h"

#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Health.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/gameplay/Behaviour.h"
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
    // S10: the object is recreated from classId by the factory (spawn messages carry the class, S16)
    registry.registerComponent<Behaviour>(Replicate::Never);
    registry.registerComponent<Persistent>(Replicate::Host);
    registry.registerComponent<Player>(Replicate::Host);
    registry.registerComponent<Health>(Replicate::Host);
    registry.registerComponent<PlayerInput>(Replicate::OwnerAuthoritative);
    registry.registerComponent<CharacterIntent>(Replicate::Never);
}

}  // namespace Engine
}  // namespace Saiga
