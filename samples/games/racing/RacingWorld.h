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
#include "games/racing/DriverModel.h"
#include "games/racing/LapState.h"
#include "games/racing/Track.h"
#include "games/racing/Vehicle.h"

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
 * The simulation of the racing tech demo (docs/engine_design.md §18.4, S13b) in a World: the car's state in components
 * (VehicleState, VehicleControls, LapState, its pose in Transform), the vehicle model and the lap timing as systems
 * (vehicleSystem, lapTimerSystem), the decisions as game objects (Car: back to the track when stuck on the roof or on
 * the respawn action; RaceDriver: the demo's AI). The terrain and the track are the context.
 */

struct RacingContext
{
    Vulkan::HeightField heightField;
    Track track;
    int treeCount = 0;  // the trees the track avoids (the renderer must use the same count and seed)
    double terrain(double x, double z) const { return heightField.height(float(x), float(z)); }
    // The car's ground: the lifted asphalt on the track, the terrain elsewhere (hint: as Track::surfaceOffset)
    double carGround(double x, double z, double& hint) const
    {
        return terrain(x, z) + track.surfaceOffset(Vec3(x, 0, z), hint);
    }
    // Generates the height field, places the trees and the track (on gentle land with few trees; the time is printed)
    static std::shared_ptr<const RacingContext> create(int treeCount);
};

// The car's state besides its pose (R-STATE-1)
struct VehicleState
{
    static constexpr const char* name       = "VehicleState";
    static constexpr Engine::ComponentId id = Engine::ComponentId(RacingComponent::VehicleState);
    static constexpr uint32_t version       = 1;
    Engine::DataRef<VehicleParameters> model{"default"};
    Vec3 velocity                     = Vec3::Zero();
    Vec3 angularVelocity              = Vec3::Zero();
    std::array<double, 4> compression = {};
    std::array<bool, 4> contact       = {};
    std::array<double, 4> spin        = {};
    double steerAngle                 = 0;
    double surfaceHint                = -1;            // the arc length near the car (speeds up the track queries)
    double upsideDown                 = 0;             // seconds on the roof
    Vec3 previousPosition             = Vec3::Zero();  // before the last step (the lap timer's gate crossing)

    template <class V>
    void visit(V& v)
    {
        v("model", model);
        v("velocity", velocity);
        v("angularVelocity", angularVelocity);
        v("compression", compression);
        v("contact", contact);
        v("spin", spin);
        v("steerAngle", steerAngle);
        v("surfaceHint", surfaceHint);
        v("upsideDown", upsideDown);
        v("previousPosition", previousPosition);
    }
};

// The AI driver's progress on the track
struct DriverState
{
    static constexpr const char* name       = "DriverState";
    static constexpr Engine::ComponentId id = Engine::ComponentId(RacingComponent::DriverState);
    static constexpr uint32_t version       = 1;
    double progress                         = -1;
    template <class V>
    void visit(V& v)
    {
        v("progress", progress);
    }
};

enum class RacingClass : uint16_t
{
    Car = Engine::kFirstGameClass,
    RaceDriver,
};
enum class RacingAction : uint32_t
{
    Respawn = 1,
};

// The vehicle model of an entity's components (pose from Transform)
Vehicle vehicleOf(const World& w, Entity e);

class CarBase : public Engine::GameObject
{
   public:
    void onSpawn() override;
    void update(double dt) override;
    const Engine::TickPolicy* tickPolicy() const override { return everyStepTickPolicy(); }
    // Back on the track at the last checkpoint
    void respawn();

   protected:
    // Standing at arc length s of the track, heading along it
    void placeOnTrack(double s);
};

// The player's car (VehicleControls from the input map)
class Car final : public CarBase
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(RacingClass::Car);
    static constexpr const char* name   = "Car";
};

// The demo's car: the AI driver races (writes VehicleControls)
class RaceDriver final : public CarBase
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(RacingClass::RaceDriver);
    static constexpr const char* name   = "RaceDriver";
    void onSpawn() override;
    void update(double dt) override;
};

Engine::WorldSetup racingWorldSetup(std::shared_ptr<const RacingContext> context);
void racingScenarios(Engine::ScenarioList& list);

}  // namespace Game
}  // namespace Saiga
