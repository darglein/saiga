/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/world/World.h"

#include "games/common/DrivingComponents.h"

#include <string>

namespace Saiga
{
namespace Engine
{
struct AppOptions;
class Registry;
}  // namespace Engine
namespace Game
{
using Engine::Entity;
using Engine::World;

/**
 * The driver-entity pattern of the tech demos (docs/engine_design.md §18.4, §13.3; S16b): vehicles stay host-owned; a
 * peer's player is a driver entity (Player, PlayerInput, the game's control component, Driver) owned by the peer, and
 * the host's vehicle system reads the controls of VehicleSeat.driver (one round trip of steering latency, §20). AI
 * vehicles are their own driver (Driver{self}, VehicleSeat{self}).
 */

// Driver, VehicleSeat (Replicate::Host) and DemoPilot (Never, saved)
void registerDrivingComponents(Engine::Registry& r);

struct Driven
{
    Entity driver, vehicle;
};

// The host's player of a peer with a vehicle of class T (T::Controls: the game's control component): the driver
// entity first (Player{peer}, PlayerInput, T::Controls, an empty RenderInstance: tracked, not drawn), then the vehicle
// (host-owned) with the initial VehicleSeat {driver, slot}, then Driver.vehicle, the driver's pose at the vehicle and
// setOwner(driver, peer) (peer 0: the host's own player, no NetOwner). Between steps, on the host; does not step.
template <class T>
Driven spawnDriven(World& w, uint8_t peer, int slot)
{
    const Entity driver = w.create();
    Engine::Player p;
    p.peer        = peer;
    p.characterId = uint64_t(peer) + 1;
    w.add<Engine::Player>(driver, p);
    w.add<Engine::PlayerInput>(driver);
    w.add<typename T::Controls>(driver);
    // Tracked by the extract, not drawn (interpolatedPose; S19b)
    w.add<Engine::RenderInstance>(driver);
    VehicleSeat seat;
    seat.driver          = driver;
    seat.slot            = slot;
    const Entity vehicle = w.spawn<T>(Vec3::Zero(), seat);
    w.add<Driver>(driver, Driver{vehicle});
    const Engine::Transform& t = w.read<Engine::Transform>(vehicle);
    w.get<Engine::Transform>(driver).teleport(t.position(), t.rotation());
    if (peer != 0) w.setOwner(driver, peer);
    return {driver, vehicle};
}

// The vehicle a driver drives (none: Entity{})
Entity drivenVehicle(const World& w, Entity driver);
// The controls of a vehicle's driver (nullptr: no live driver, the vehicle gets neutral controls)
template <class C>
const C* seatControls(const World& w, Entity vehicle)
{
    if (!w.has<VehicleSeat>(vehicle)) return nullptr;
    const Entity d = w.read<VehicleSeat>(vehicle).driver;
    if (!d.valid() || !w.alive(d) || !w.has<C>(d)) return nullptr;
    return &w.read<C>(d);
}
// Whether the action's player (a driver) drives this vehicle (actions name the driver; AI vehicles drive themselves)
bool drives(const World& w, Entity player, Entity vehicle);

// The leave policy (R-MECH-1): the peer's driver (all its owned entities, destroyOwnedBy) is removed and its vehicle
// stays empty (VehicleSeat.driver cleared: neutral controls). Host, between steps
void removeDriver(World& host, uint8_t peer);

// Phase Gameplay after the vehicle system, RunOn::Both over ownedBy(localPeer): the vehicle's pose into the driver's
// Transform (the host: the vehicle's Transform; a client: its RenderPose), so tools, the grid and interest see the
// player at its vehicle. Plain setPose (nothing renders the driver)
void driverPoseSystem(World& w, double dt);

// --coop --demo: the demo pilots' gains stay stable up to a delay of 8 steps (EngineAppSetup::checkOptions)
bool clampCoopDemoDelay(Engine::AppOptions& options, std::string& error);

}  // namespace Game
}  // namespace Saiga
