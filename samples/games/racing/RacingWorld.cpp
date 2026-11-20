/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The racing demo in a World: components, the vehicle and lap timer systems, Car and RaceDriver, scenarios.

#include "games/racing/RacingWorld.h"

#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"
#include "saiga/vulkan/world/FoliageGenerator.h"
#include "saiga/vulkan/world/ForestRenderer.h"

#include "games/common/RenderAssets.h"

#include <chrono>
#include <cstdio>
#include <iostream>

namespace Saiga
{
namespace Game
{
using namespace Engine;

std::shared_ptr<const RacingContext> RacingContext::create()
{
    auto c        = std::make_shared<RacingContext>();
    const auto t0 = std::chrono::steady_clock::now();
    c->heightField.generate(Vulkan::HeightField::Parameters());
    const Vulkan::HeightField& hf = c->heightField;

    // A site on gentle land with few trees on the track (the prototype's trees cannot be removed): the trees of the
    // world prototype (WorldParameters::seed 1) per 16 m cell
    const auto trees =
        Vulkan::ForestRenderer::placeTrees(hf, kSiteTreeCount, 1, Vulkan::FoliageGenerator::treeSpecies(1));
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

// Phase Gameplay (host): the vehicle model of every car (after the AI wrote its controls in phase Behaviour), with the
// controls of its seat's driver (a client's driver: the latest received; no live driver: neutral)
void vehicleSystem(World& w, double dt)
{
    const RacingContext* ctx = &w.context<RacingContext>();
    w.view<Transform, VehicleState>().each(
        [&](Entity e, Transform& t, VehicleState& s)
        {
            const VehicleControls* driven = seatControls<VehicleControls>(w, e);
            const VehicleControls c       = driven ? *driven : VehicleControls();
            double hint                   = s.surfaceHint;
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

// Phase Gameplay on clients (--coop --demo): the demo's AI drives the owned drivers with DemoPilot from the replicated
// car (VehicleState and the newest snapshot pose); its progress restarts when the car was teleported (a respawn)
void racingDemoPilotSystem(World& w, double)
{
    const Track& track = w.context<RacingContext>().track;
    w.ownedBy<VehicleControls, DemoPilot, const Driver>(w.localPeer())
        .each(
            [&](Entity, VehicleControls& c, DemoPilot& pilot, const Driver& d)
            {
                if (pilot.suspended || !d.vehicle.valid() || !w.alive(d.vehicle) || !w.has<VehicleState>(d.vehicle))
                    return;
                if (w.read<Transform>(d.vehicle).wasTeleported()) pilot.progress = -1;
                DriverModel driver;
                driver.progress = pilot.progress;
                c               = driver.controls(vehicleOf(w, d.vehicle), track);
                pilot.progress  = driver.progress;
            });
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
    // The body is drawn by the engine (S19b; the wheels are the renderer's cosmetic instances)
    if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance(kCarBodyMesh));
    if (!has<VehicleSeat>())
    {
        // Its own driver: an AI car, or a car spawned as the player
        add<VehicleSeat>(VehicleSeat{entity(), 0});
        add<Driver>(Driver{entity()});
        if (!has<VehicleControls>()) add<VehicleControls>();
    }
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
    car.params = world().data(read<VehicleState>().model);
    car.ground = [ctx, &hint](double x, double z) { return ctx->carGround(x, z, hint); };
    // The slot's place in the grid: 8 m behind per row, 3 m left or right of the center line
    const int slot = has<VehicleSeat>() ? std::max(0, read<VehicleSeat>().slot) : 0;
    if (slot > 0) s -= 8.0 * ((slot + 1) / 2);
    const Vec3 tangent = ctx->track.tangent(s);
    const Vec3 right(-tangent(2), 0, tangent(0));
    const Vec3 p = ctx->track.position(s) + right * (slot == 0 ? 0.0 : (slot % 2 ? 3.0 : -3.0));
    car.resetOnGround(p(0), p(2), headingOf(tangent));
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
    const int next     = LapState::wrapCheckpoint(read<LapState>().nextCheckpoint, track);
    const int last     = (next - 1 + int(gates.size())) % int(gates.size());
    placeOnTrack(gates[last].s + 3);
}

void CarBase::update(double)
{
    // The action's player is the driver (the AI car and a car spawned as the player drive themselves)
    bool requested = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (drives(world(), a.player, entity()) && a.action == uint32_t(RacingAction::Respawn)) requested = true;
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

void racingPeerJoined(World& host, uint8_t peer, bool demo)
{
    const Driven d = spawnDriven<Car>(host, peer, peer);
    if (demo) host.add<DemoPilot>(d.driver);
}

WorldSetup racingWorldSetup(std::shared_ptr<const RacingContext> context)
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<VehicleControls>(Replicate::OwnerAuthoritative);
        r.registerComponent<VehicleState>(Replicate::Host);
        r.registerComponent<LapState>(Replicate::Host);
        r.registerComponent<DriverState>(Replicate::Never);
        registerDrivingComponents(r);
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "racingDemoPilot", racingDemoPilotSystem, RunOn::Client);
        l.add(Phase::Gameplay, "vehicle", vehicleSystem);
        l.add(Phase::Gameplay, "lapTimer", lapTimerSystem);
        l.add(Phase::Gameplay, "driverPose", driverPoseSystem, RunOn::Both);
    };
    s.classes = [](ClassList& c)
    {
        c.add<Car>();
        c.add<RaceDriver>();
    };
    s.actions      = [](ActionList& a) { a.add(uint32_t(RacingAction::Respawn), "respawn"); };
    s.dataTypes    = [](DataTypeList& d) { d.add<VehicleParameters>(); };
    s.renderAssets = [](World& w)
    {
        const Vulkan::MaterialId flat = w.registerMaterial("racing/flat", demoMaterial());
        registerDemoMesh(w, kCarBodyMesh, []() { return MeshBuilder::carBody(Vec3(0.85, 0.15, 0.12)); }, flat);
        // One wheel mesh per radius of the vehicle models (no non-uniform scale on instances)
        for (const std::string& model : w.dataNames<VehicleParameters>())
        {
            const double r = w.data<VehicleParameters>(model).wheelRadius;
            if (w.findMesh(wheelMeshName(r))) continue;
            registerDemoMesh(w, wheelMeshName(r), [r]() { return MeshBuilder::wheel(r, 0.28); }, flat);
        }
    };
    s.setContext(std::move(context));
    return s;
}

std::string wheelMeshName(double radius)
{
    char name[64];
    std::snprintf(name, sizeof(name), "racing/wheel/r%.4f", radius);
    return name;
}

// ---- scenarios -----------------------------------------------------------------------------------------------------

void racingScenarios(ScenarioList& list)
{
    // S16b: a driver of the default peer (a client's in the host + client variants), the controls written on its owner;
    // a progress window instead of a fixed 3 s (the controls arrive up to 8 steps late)
    list.add(
        "accelerates_on_track",
        [](ScenarioWorld& w)
        {
            const Driven d   = spawnDriven<Car>(w.world(), w.defaultPeer(), 0);
            const Entity car = d.vehicle;
            w.edit(d.driver, [](World& owner, Entity e) { owner.get<VehicleControls>(e) = VehicleControls{1.0, 0.0}; });
            w.trace<VehicleState>(car);
            const Track& track = w.world().context<RacingContext>().track;
            w.check(w.stepUntil([&] { return track.closest(w.world().read<Transform>(car).position()) > 25; }, 60 * 6),
                    "reaches 25 m along the track");
            const Vehicle v = vehicleOf(w.world(), car);
            // Straight ahead from the start line (the track curves away after a few seconds)
            const double progress = track.closest(v.state.position);
            w.check(v.forwardSpeed() > 12, "accelerates: " + std::to_string(v.forwardSpeed()) + " m/s");
            w.check(progress > 15 && progress < 40, "along the track: " + std::to_string(progress) + " m");
            w.check(v.wheelsOnGround() == 4, "all wheels on the ground");
            // The track of the default world, independent of SAIGA_WORLD_TREES (the renderer's tree count)
            w.check(std::abs(track.length() - 1479.23) < 0.01,
                    "the default track: " + std::to_string(track.length()) + " m");
        });
    // Regression: a nextCheckpoint set from outside (set_component) indexed the gates out of range (an abort, or
    // undefined behavior for a negative value); it wraps around. S16b: the car spawned as the host's player is its own
    // driver (onSpawn adds the links); the LapState edit is on the host (Host component)
    list.add("checkpoint_out_of_range",
             [](ScenarioWorld& w)
             {
                 const Entity car = w.spawnPlayer<Car>(Vec3::Zero(), 0);
                 for (int bad : {100000000, -5})
                 {
                     w.world().get<LapState>(car).nextCheckpoint = bad;
                     w.input(car, uint32_t(RacingAction::Respawn));
                     w.step(3);
                     const int n = w.world().read<LapState>(car).nextCheckpoint;
                     w.check(n >= 0 && n < 10, "in range after " + std::to_string(bad) + ": " + std::to_string(n));
                 }
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
