/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/world/Phase.h"
#include "saiga/export.h"

#include <array>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

// A system: a free function (R-SYS-1)
using SystemFn = void (*)(World& world, double dt);

/**
 * The systems of a world per phase (docs/engine_design.md §8.2). They run in registration order inside their phase;
 * the engine's list is registered first (registerEngineSystems), then the game's (WorldSetup::systems). Names are
 * unique. RunOn: before S16 the world is the host, so Host and Both systems run.
 */
class SAIGA_ENGINE_API SystemList
{
   public:
    struct System
    {
        Phase phase;
        std::string name;
        SystemFn fn;
        RunOn runOn;
    };

    void add(Phase phase, const char* name, SystemFn fn, RunOn runOn = RunOn::Host);

    const std::vector<System>& systems(Phase phase) const { return perPhase[size_t(phase)]; }
    size_t size() const;

   private:
    std::array<std::vector<System>, size_t(kLastPhase) + 1> perPhase;
};

}  // namespace Engine
}  // namespace Saiga
