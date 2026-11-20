/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The racing demo in a World: components, the vehicle and lap timer systems, Car and RaceDriver, scenarios.

#include "games/racing/RacingWorld.h"

#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/ForestRenderer.h"

#include <chrono>
#include <iostream>

namespace Saiga
{
namespace Game
{
using namespace Engine;

std::shared_ptr<const RacingContext> RacingContext::create(int treeCount)
{
    auto c        = std::make_shared<RacingContext>();
    const auto t0 = std::chrono::steady_clock::now();
    c->treeCount  = treeCount;
    c->heightField.generate(Vulkan::HeightField::Parameters());
    const Vulkan::HeightField& hf = c->heightField;

    // A site on gentle land with few trees on the track (the prototype's trees cannot be removed): the trees of the
    // world prototype (WorldParameters::seed 1) per 16 m cell
    const auto trees  = Vulkan::ForestRenderer::placeTrees(hf, treeCount, 1, Vulkan::FoliageGenerator::treeSpecies(1));
    const double cell = 16;
    const int cells   = int(hf.worldSize() / cell) + 1;
    std::vector<int> perCell(size_t(cells) * cells, 0);
    for (auto& t : trees)
    {
        int cx = int(t.posYaw(0) / cell), cz = int(t.posYaw(2) / cell);
        if (cx >= 0 && cz >= 0 && cx < cells && cz < cells) perCell[size_t(cz) * cells + cx]++;
    }
    auto treePenalty = [&](const Vec3& p)
    {
        int cx = int(p(0) / cell), cz = int(p(2) / cell);
        return (cx >= 0 && cz >= 0 && cx < cells && cz < cells) ? 3.0 * perCell[size_t(cz) * cells + cx] : 0.0;
    };
    const RacingContext* ctx = c.get();
    GroundHeight ground      = [ctx](double x, double z) { return ctx->terrain(x, z); };
    const double radius      = 220;
    const uint32_t seed      = 7;
    const Vec3 site = Track::findSite(ground, hf.worldSize(), radius, hf.waterLevel() + 3, 12, seed, treePenalty);
    c->track.width  = 12;
    c->track.build(Track::loop(site, radius, 12, seed), 10);
    std::cout << "Racing: height field, " << trees.size() << " trees and the track (" << c->track.length() << " m) in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
    return c;
}

namespace
{
VehicleKinematics kinematicsOf(const Transform& t, const VehicleState& s)
{
    VehicleKinematics k;
    k.position        = t.position();
    k.orientation     = t.rotation();
    k.velocity        = s.velocity;
    k.angularVelocity = s.angularVelocity;
    k.compression     = s.compression;
    k.contact         = s.contact;
    k.spin            = s.spin;
    k.steerAngle      = s.steerAngle;
    return k;
}

void store(const VehicleKinematics& k, Transform& t, VehicleState& s, bool teleport)
{
    if (teleport)
        t.teleport(k.position, k.orientation);
    else
    {
        t.setPosition(k.position);
        t.setRotation(k.orientation);
    }
    s.velocity        = k.velocity;
    s.angularVelocity = k.angularVelocity;
    s.compression     = k.compression;
    s.contact         = k.contact;
    s.spin            = k.spin;
    s.steerAngle      = k.steerAngle;
}

// Phase Gameplay: the vehicle model of every car (after the drivers wrote the controls in phase Behaviour)
void vehicleSystem(World& w, double dt)
{
    const RacingContext* ctx = &w.context<RacingContext>();
    w.view<Transform, VehicleState, const VehicleControls>().each(
        [&](Entity, Transform& t, VehicleState& s, const VehicleControls& c)
        {
            double hint = s.surfaceHint;
            Vehicle car;
            car.params         = w.data(s.model);
            car.state          = kinematicsOf(t, s);
            car.ground         = [ctx, &hint](double x, double z) { return ctx->carGround(x, z, hint); };
            s.previousPosition = t.position();
            car.step(c, dt);
            store(car.state, t, s, false);
            s.surfaceHint = hint;
            // Stuck on the roof: the Car game object puts it back on the track after 2 s
            s.upsideDown = car.up()(1) < 0.3 ? s.upsideDown + dt : 0;
        });
}

// Phase Gameplay, after vehicleSystem: the gates the cars crossed in this step
void lapTimerSystem(World& w, double dt)
{
    const Track& track = w.context<RacingContext>().track;
    w.view<const Transform, const VehicleState, LapState>().each(
        [&](Entity, const Transform& t, const VehicleState& s, LapState& lap)
        { lap.update(track, s.previousPosition, t.position(), dt); });
}
}  // namespace

Vehicle vehicleOf(const World& w, Entity e)
{
    const RacingContext* ctx = &w.context<RacingContext>();
    const VehicleState& s    = w.read<VehicleState>(e);
    Vehicle car;
    car.params = w.data(s.model);
    car.state  = kinematicsOf(w.read<Transform>(e), s);
    car.ground = [ctx](double x, double z)
    {
        double hint = -1;
        return ctx->carGround(x, z, hint);
    };
    return car;
}

void CarBase::onSpawn()
{
    if (!has<VehicleState>()) add<VehicleState>();
    if (!has<VehicleControls>()) add<VehicleControls>();
    if (!has<LapState>()) add<LapState>();
    write<LapState>().reset(world().context<RacingContext>().track);
    placeOnTrack(0);
}

void CarBase::placeOnTrack(double s)
{
    const RacingContext* ctx = &world().context<RacingContext>();
    // Regression: the track search starts at the hint; a stale hint from the crash site missed the asphalt lift and the
    // car spawned 0.15 m too low (it hopped)
    double hint = -1;
    Vehicle car;
    car.params   = world().data(read<VehicleState>().model);
    car.ground   = [ctx, &hint](double x, double z) { return ctx->carGround(x, z, hint); };
    const Vec3 p = ctx->track.position(s);
    car.resetOnGround(p(0), p(2), headingOf(ctx->track.tangent(s)));
    VehicleState& state = write<VehicleState>();
    store(car.state, transformWrite(), state, true);
    state.surfaceHint      = hint;
    state.upsideDown       = 0;
    state.previousPosition = car.state.position;
    if (has<DriverState>()) write<DriverState>().progress = -1;
}

void CarBase::respawn()
{
    const Track& track = world().context<RacingContext>().track;
    const auto& gates  = track.checkpoints();
    const int last     = (read<LapState>().nextCheckpoint - 1 + int(gates.size())) % int(gates.size());
    placeOnTrack(gates[last].s + 3);
}

void CarBase::update(double)
{
    bool requested = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (a.player == entity() && a.action == uint32_t(RacingAction::Respawn)) requested = true;
    if (requested || read<VehicleState>().upsideDown > 2.0) respawn();
}

void RaceDriver::onSpawn()
{
    if (!has<DriverState>()) add<DriverState>();
    CarBase::onSpawn();
}

void RaceDriver::update(double dt)
{
    CarBase::update(dt);
    DriverModel driver;
    driver.progress          = read<DriverState>().progress;
    write<VehicleControls>() = driver.controls(vehicleOf(world(), entity()), world().context<RacingContext>().track);
    write<DriverState>().progress = driver.progress;
}

WorldSetup racingWorldSetup(std::shared_ptr<const RacingContext> context)
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<VehicleControls>();
        r.registerComponent<VehicleState>();
        r.registerComponent<LapState>();
        r.registerComponent<DriverState>();
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "vehicle", vehicleSystem);
        l.add(Phase::Gameplay, "lapTimer", lapTimerSystem);
    };
    s.classes = [](ClassList& c)
    {
        c.add<Car>();
        c.add<RaceDriver>();
    };
    s.actions   = [](ActionList& a) { a.add(uint32_t(RacingAction::Respawn), "respawn"); };
    s.dataTypes = [](DataTypeList& d) { d.add<VehicleParameters>(); };
    s.setContext(std::move(context));
    return s;
}

// ---- scenarios -----------------------------------------------------------------------------------------------------

void racingScenarios(ScenarioList& list)
{
    list.add("accelerates_on_track",
             [](ScenarioWorld& w)
             {
                 const Entity car                    = w.spawnPlayer<Car>(Vec3::Zero(), 0);
                 w.world().get<VehicleControls>(car) = VehicleControls{1.0, 0.0};
                 w.trace<VehicleState>(car);
                 w.stepSeconds(3);
                 const Vehicle v    = vehicleOf(w.world(), car);
                 const Track& track = w.world().context<RacingContext>().track;
                 // Straight ahead from the start line (the track curves away after a few seconds)
                 const double progress = track.closest(v.state.position);
                 w.check(v.forwardSpeed() > 12, "accelerates: " + std::to_string(v.forwardSpeed()) + " m/s");
                 w.check(progress > 15 && progress < 40, "along the track: " + std::to_string(progress) + " m");
                 w.check(v.wheelsOnGround() == 4, "all wheels on the ground");
             });
    // The behavior before the port to the engine (S13b): the AI's first lap takes 62.07 s
    list.add("ai_completes_a_lap",
             [](ScenarioWorld& w)
             {
                 const Entity car = w.spawn<RaceDriver>(Vec3::Zero());
                 w.trace<LapState>(car);
                 w.check(w.stepUntil([&] { return w.world().read<LapState>(car).laps >= 1; }, 60 * 90),
                         "completes a lap");
                 const double t = w.world().read<LapState>(car).lastLapTime;
                 w.check(std::abs(t - 62.07) < 0.5, "in 62.07 s: " + std::to_string(t));
             });
}

}  // namespace Game
}  // namespace Saiga
