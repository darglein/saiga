/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The tech demos' driver entities (docs/engine_design.md §18.4, S16b): Driver, VehicleSeat and DemoPilot.

#pragma once

#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"

#include <cstdint>

namespace Saiga
{
namespace Game
{
// The components of samples/games/common, the same IDs in every demo (registration IDs are dense: the demos' own
// enums start at kFirstDemoComponent)
enum class CommonComponent : uint8_t
{
    Driver = Engine::kFirstGameComponent,
    VehicleSeat,
    DemoPilot,
};
constexpr Engine::ComponentId kFirstDemoComponent = Engine::ComponentId(CommonComponent::DemoPilot) + 1;

// On a driver entity (the player of a peer, or an AI vehicle itself): the vehicle it drives. Replicate::Host, never
// written by the owner (§13.3: vehicles stay host-owned; the driver's owner writes only its controls)
struct Driver
{
    static constexpr const char* name       = "Driver";
    static constexpr Engine::ComponentId id = Engine::ComponentId(CommonComponent::Driver);
    static constexpr uint32_t version       = 1;
    Engine::Entity vehicle;

    template <class V>
    void visit(V& v)
    {
        v("vehicle", vehicle);
    }
};

// On a vehicle: its driver (the reverse link the vehicle systems read the controls through; none or dead: neutral
// controls) and its slot (a grid offset of the start and respawn positions, kept by respawns). Replicate::Host
struct VehicleSeat
{
    static constexpr const char* name       = "VehicleSeat";
    static constexpr Engine::ComponentId id = Engine::ComponentId(CommonComponent::VehicleSeat);
    static constexpr uint32_t version       = 1;
    Engine::Entity driver;
    int32_t slot = 0;

    template <class V>
    void visit(V& v)
    {
        v("driver", driver);
        v("slot", slot);
    }
};

// A driver flown / driven / run by the demo's pilot on its owner (--coop --demo): the owner-side pilot systems
// (RunOn::Client) write its controls. suspended: device input is active (the player took over). progress (racing)
// and waypoint (platformer; -1: the restart after the goal was requested): the pilot's small state, reset when the
// vehicle is teleported. Replicate::Never, saved
// (it reaches the owner through the spawn's presence list)
struct DemoPilot
{
    static constexpr const char* name       = "DemoPilot";
    static constexpr Engine::ComponentId id = Engine::ComponentId(CommonComponent::DemoPilot);
    static constexpr uint32_t version       = 1;
    bool suspended                          = false;
    double progress                         = -1;
    int32_t waypoint                        = 0;

    template <class V>
    void visit(V& v)
    {
        v("suspended", suspended);
        v("progress", progress);
        v("waypoint", waypoint);
    }
};

}  // namespace Game
}  // namespace Saiga
