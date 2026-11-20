/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/data/DataRef.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/vulkan/world/HeightField.h"

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
 */

// Constant configuration (World::context): the world prototype's height field and the autopilot's circuit
struct FlightContext
{
    Vulkan::HeightField heightField;
    Vec3 circuitCenter   = Vec3::Zero();
    double circuitRadius = 2500;
    // Terrain or water, whichever is higher (the plane crashes into both)
    double ground(double x, double z) const
    {
        return std::max(double(heightField.height(float(x), float(z))), double(heightField.waterLevel()));
    }
    // Generates the height field (default parameters of the world prototype; the time is printed)
    static std::shared_ptr<const FlightContext> create();
};

// The airplane's state besides its pose (R-STATE-1)
struct AircraftState
{
    static constexpr const char* name       = "AircraftState";
    static constexpr Engine::ComponentId id = Engine::ComponentId(FlightComponent::AircraftState);
    static constexpr uint32_t version       = 1;
    Engine::DataRef<AircraftParameters> model{"default"};
    Vec3 velocity     = Vec3::Zero();
    Vec3 bodyRates    = Vec3::Zero();
    bool onGround     = false;
    bool crashed      = false;
    double crashTimer = 0;  // seconds since the crash

    template <class V>
    void visit(V& v)
    {
        v("model", model);
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
    void onSpawn() override;
    void update(double dt) override;
    const Engine::TickPolicy* tickPolicy() const override { return everyStepTickPolicy(); }
    // On the circuit, heading along it, 300 m above the ground, at 65 m/s
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

// The world setup of the flight demo (components, systems, classes, actions, data, the context)
Engine::WorldSetup flightWorldSetup(std::shared_ptr<const FlightContext> context, const std::string& dataDirectory);
void flightScenarios(Engine::ScenarioList& list);

}  // namespace Game
}  // namespace Saiga
