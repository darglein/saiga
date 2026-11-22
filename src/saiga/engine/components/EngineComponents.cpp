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
#include "saiga/engine/components/NetLocal.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void registerEngineComponents(Registry& registry, const WorldSetup& setup)
{
    registry.registerComponent<Parent>(Replicate::Host);
    registry.registerComponent<LocalTransform>(Replicate::Host);
    // S10: the object is recreated from classId by the factory (spawn messages carry the class, S16)
    registry.registerComponent<Behaviour>(Replicate::Never);
    registry.registerComponent<Persistent>(Replicate::Host);
    registry.registerComponent<Player>(Replicate::Host);
    registry.registerComponent<Health>(Replicate::Host);
    registry.registerComponent<PlayerInput>(Replicate::OwnerAuthoritative);
    // Computed on each peer; a client default-constructs it from the spawn's presence list (§13.4)
    registry.registerComponent<CharacterIntent>(Replicate::Never);
    // S16: ownership, host-local entities, the interpolated render pose
    registry.registerComponent<NetOwner>(Replicate::Host);
    registry.registerComponent<NetLocal>(Replicate::Never);
    registry.registerComponent<RenderPose>(Replicate::Never, Saved::No);
    // S19b: what an entity renders, by mesh name (rarely changing: reliable, not in the snapshots)
    registry.registerComponent<RenderInstance>(Replicate::HostReliable);
    // S21: entrances (clients show prompts) and requested transitions (host-only state, saved)
    registry.registerComponent<SpaceLink>(Replicate::Host);
    registry.registerComponent<PendingTransition>(Replicate::Never);
    // S36: the time of day and the weather; world and space state only with the environment (D4), so plain worlds
    // keep their entities, hashes and saves
    if (setup.environment.enabled)
    {
        registry.registerGlobal<WorldTime>(Replicate::Host);
        registry.registerSpaceState<Weather>(Replicate::Host);
    }
    else
    {
        registry.registerComponent<WorldTime>(Replicate::Host);
        registry.registerComponent<Weather>(Replicate::Host);
    }
}

}  // namespace Engine
}  // namespace Saiga
