/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/data/DataRef.h"
#include "saiga/engine/gameplay/GameObject.h"

#include "games/common/Driving.h"
#include "games/common/GameEngine.h"
#include "games/flight/FlightAutopilot.h"
#include "games/flight/FlightModel.h"

#include <memory>

namespace Saiga
{
namespace Engine
{
class ScenarioList;
}
namespace Game
{
/**
 * The simulation of the flight simulator tech demo (docs/engine_design.md §18.4, S13b) in a World: the airplane's
 * state in components (AircraftState, AircraftControls, its pose in Transform), the flight model as a system
 * (flightModelSystem, phase Gameplay), the decisions as game objects (Aircraft: respawn after a crash; Autopilot: the
 * demo's pilot). The ground is the context (FlightContext), shared by all worlds of the process and the renderer.
 * S16b: the planes are host-owned; a player is a driver entity (AircraftControls: the sticks and throttleChange,
 * Driver) that the plane's VehicleSeat names (games/common/Driving.h); the autopilot's plane is its own driver and sets
 * the throttle lever directly. Replication: AircraftControls OwnerAuthoritative (the driver's owner), AircraftState
 * (with the lever) Host.
 */

// Constant configuration (World::context): the world seed and the autopilot's circuit (S23c after the live demo: in the
// centre of the 32 km space, kFlightCircuitCentre, with land under it; S20 had it in the corner). The context is
// World-independent and shared by the co-op worlds and the renderer; the ground is the world's terrain (flightGround)
struct FlightContext
{
    // The space seed (WorldSetup::seed)
    uint64_t seed        = 1;
    Vec3 circuitCenter   = Vec3::Zero();
    double circuitRadius = 2500;
    // The context of the world seed: the circuit around (16384, 16384) with S20's radius 0.3 * 8192 m
    static std::shared_ptr<const FlightContext> create(uint64_t seed = 1);
};

// The ground the plane crashes into: the world's terrain of space 0 (World::terrainHeight: the generated heights with
// the sites' patches) or the sea, whichever is higher (S23c, replacing S20's FlightContext::ground of the hook)
double flightGround(const World& world, double x, double z);

// The airplane's state besides its pose (R-STATE-1)
struct AircraftState
{
    static constexpr const char* name       = "AircraftState";
    static constexpr Engine::ComponentId id = Engine::ComponentId(FlightComponent::AircraftState);
    static constexpr uint32_t version       = 2;  // S16b: the throttle lever moved here from AircraftControls
    Engine::DataRef<AircraftParameters> model{"default"};
    double throttle   = 0.6;  // the lever [0, 1]: moved by the driver's throttleChange, set by the autopilot
    Vec3 velocity     = Vec3::Zero();
    Vec3 bodyRates    = Vec3::Zero();
    bool onGround     = false;
    bool crashed      = false;
    double crashTimer = 0;  // seconds since the crash

    template <class V>
    void visit(V& v)
    {
        v("model", model);
        v("throttle", throttle);
        v("velocity", velocity);
        v("bodyRates", bodyRates);
        v("onGround", onGround);
        v("crashed", crashed);
        v("crashTimer", crashTimer);
    }
};

enum class FlightClass : uint16_t
{
    Aircraft = Engine::kFirstGameClass,
    Autopilot,
    FlightChest,
};

// The player actions of the flight demo
enum class FlightAction : uint32_t
{
    Respawn = 1,
};

// The flight model of an entity's components (pose from Transform), for the autopilot, the GUI and the system
FlightModel flightModelOf(const World& w, Entity e);

// The plane's decisions: respawn on the autopilot's circuit after a crash (2 s) or on the respawn action
class AircraftBase : public Engine::GameObject
{
   public:
    // The control component of its driver (spawnDriven)
    using Controls = AircraftControls;
    // Without an initial VehicleSeat (the autopilot, a plane spawned as the player) the plane is its own driver
    void onSpawn() override;
    void update(double dt) override;
    const Engine::TickPolicy* tickPolicy() const override { return everyStepTickPolicy(); }
    // On the circuit, heading along it, 300 m above the ground, at 65 m/s (offset by the slot of VehicleSeat: behind
    // and inside of the circuit)
    void respawn();
};

// The player's plane (its controls come from the input map, EngineAppSetup::applyInput)
class Aircraft final : public AircraftBase
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(FlightClass::Aircraft);
    static constexpr const char* name   = "Aircraft";
};

// The demo's plane: the autopilot flies the circuit (writes AircraftControls)
class Autopilot final : public AircraftBase
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(FlightClass::Autopilot);
    static constexpr const char* name   = "Autopilot";
    void configure() override;
    void update(double dt) override;

   private:
    FlightAutopilot pilot;  // constant configuration (configure)
};

// The chest of the ruin prefab (data/flight/prefabs/ruin.json, S23c): stamped around players; its Health comes from
// the prefab
class FlightChest final : public Engine::GameObject
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(FlightClass::FlightChest);
    static constexpr const char* name   = "FlightChest";
    void onSpawn() override;
};

// The airplane's mesh (WorldSetup::renderAssets; RenderInstance of every plane)
constexpr const char* kAirplaneMesh = "flight/airplane";
// The chest's mesh (its base at the origin)
constexpr const char* kChestMesh = "flight/chest";
// The foliage seed of the tree meshes (renderAssets: FoliageGenerator::treeSpecies) and of the foliage texture
// (FlightRender's PrototypeScene: setFoliageTexture), so the meshes' leaf layers match the texture
constexpr uint32_t kFlightFoliageSeed = 1;

// The flight demo's generator ID and space template (S23c): flight_overworld (OverworldGenerator), open world, 128
// chunks (32 km) at base spacing 1, the generator's sea level, fullRadius 3000 m
constexpr Engine::GeneratorId kFlightOverworldGenerator = 1;
Engine::SpaceTemplate flightSpaceTemplate();

// The world setup of the flight demo (components, systems, classes, actions, data, generator, the context)
Engine::WorldSetup flightWorldSetup(std::shared_ptr<const FlightContext> context, const std::string& dataDirectory);
void flightScenarios(Engine::ScenarioList& list);
// --coop's join policy: a plane for the joined peer (slot = peer), flown by the autopilot on the owner with demo
void flightPeerJoined(World& host, uint8_t peer, bool demo);

}  // namespace Game
}  // namespace Saiga
