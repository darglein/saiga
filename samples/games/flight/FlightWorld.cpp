/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The flight simulator demo in a World: components, the flight model system, Aircraft and Autopilot, scenarios.

#include "games/flight/FlightWorld.h"

#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"

#include "games/common/PrototypeSpace.h"
#include "games/common/RenderAssets.h"

#include <chrono>
#include <iostream>

namespace Saiga
{
namespace Game
{
using namespace Engine;

std::shared_ptr<const FlightContext> FlightContext::create(uint64_t seed)
{
    auto c            = std::make_shared<FlightContext>();
    c->seed           = seed;
    c->baseSpacing    = prototypeSpaceTemplate().baseSpacing;
    const double size = Vulkan::kPrototypeArea;
    c->circuitCenter  = Vec3(size / 2, 0, size / 2);
    c->circuitRadius  = size * 0.3;
    return c;
}

FlightModel flightModelOf(const World& w, Entity e)
{
    const FlightContext* ctx = &w.context<FlightContext>();
    const AircraftState& s   = w.read<AircraftState>(e);
    const Transform& t       = w.read<Transform>(e);
    FlightModel m;
    m.params            = w.data(s.model);
    m.ground            = [ctx](double x, double z) { return ctx->ground(x, z); };
    m.state.position    = t.position();
    m.state.orientation = t.rotation();
    m.state.velocity    = s.velocity;
    m.state.bodyRates   = s.bodyRates;
    m.state.onGround    = s.onGround;
    m.state.crashed     = s.crashed;
    return m;
}

namespace
{
// Phase Gameplay (host): the flight model of every airplane (after the autopilot wrote its controls in phase
// Behaviour), with the controls of its seat's driver (a client's driver: the latest received; no live driver: neutral);
// the driver's throttleChange moves the lever
void flightModelSystem(World& w, double dt)
{
    w.view<Transform, AircraftState>().each(
        [&](Entity e, Transform& t, AircraftState& s)
        {
            if (s.crashed)
            {
                s.crashTimer += dt;
                return;
            }
            const AircraftControls* driven = seatControls<AircraftControls>(w, e);
            const AircraftControls c       = driven ? *driven : AircraftControls();
            s.throttle                     = clampd(s.throttle + c.throttleChange * 0.5 * dt, 0, 1);
            FlightModel m                  = flightModelOf(w, e);
            m.step(c, s.throttle, dt);
            t.setPosition(m.state.position);
            t.setRotation(m.state.orientation);
            s.velocity  = m.state.velocity;
            s.bodyRates = m.state.bodyRates;
            s.onGround  = m.state.onGround;
            s.crashed   = m.state.crashed;
        });
}

// Phase Gameplay on clients (--coop --demo): the autopilot flies the owned drivers with DemoPilot from the replicated
// plane (AircraftState with the lever, the newest snapshot pose); the lever follows through throttleChange
void flightDemoPilotSystem(World& w, double)
{
    const FlightContext& ctx = w.context<FlightContext>();
    FlightAutopilot pilot;
    pilot.center = ctx.circuitCenter;
    pilot.radius = ctx.circuitRadius;
    w.ownedBy<AircraftControls, const DemoPilot, const Driver>(w.localPeer())
        .each(
            [&](Entity, AircraftControls& c, const DemoPilot& demo, const Driver& d)
            {
                if (demo.suspended || !d.vehicle.valid() || !w.alive(d.vehicle) || !w.has<AircraftState>(d.vehicle))
                    return;
                const AircraftState& s = w.read<AircraftState>(d.vehicle);
                if (s.crashed)
                {
                    c = AircraftControls();
                    return;
                }
                const FlightModel m = flightModelOf(w, d.vehicle);
                c                   = pilot.controls(m);
                c.throttleChange    = clampd((pilot.throttle(m) - s.throttle) * 4.0, -1, 1);
            });
}
}  // namespace

void AircraftBase::onSpawn()
{
    if (!has<AircraftState>()) add<AircraftState>();
    // Drawn by the engine (S19b)
    if (!has<RenderInstance>()) add<RenderInstance>(RenderInstance(kAirplaneMesh));
    if (!has<VehicleSeat>())
    {
        // Its own driver: the autopilot, or a plane spawned as the player
        add<VehicleSeat>(VehicleSeat{entity(), 0});
        add<Driver>(Driver{entity()});
        if (!has<AircraftControls>()) add<AircraftControls>();
    }
    respawn();
}

void AircraftBase::update(double)
{
    // The action's player is the driver (the autopilot and a plane spawned as the player drive themselves)
    bool requested = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (drives(world(), a.player, entity()) && a.action == uint32_t(FlightAction::Respawn)) requested = true;
    if (requested || (read<AircraftState>().crashed && read<AircraftState>().crashTimer > 2.0)) respawn();
}

void AircraftBase::respawn()
{
    const FlightContext& ctx = world().context<FlightContext>();
    // The slot's place: 60 m behind and 20 m inside per slot (heading +z on the circuit's +x side)
    const int slot = has<VehicleSeat>() ? std::max(0, read<VehicleSeat>().slot) : 0;
    Vec3 p         = ctx.circuitCenter + Vec3(ctx.circuitRadius - 20.0 * slot, 0, -60.0 * slot);
    p(1)           = ctx.ground(p(0), p(2)) + 300;
    FlightModel m;
    m.reset(p, headingOf(Vec3(0, 0, 1)), 65);
    AircraftState& s = write<AircraftState>();
    s.velocity       = m.state.velocity;
    s.bodyRates      = m.state.bodyRates;
    s.onGround       = m.state.onGround;
    s.crashed        = false;
    s.crashTimer     = 0;
    transformWrite().teleport(m.state.position, m.state.orientation);
}

void Autopilot::configure()
{
    const FlightContext& ctx = world().context<FlightContext>();
    pilot.center             = ctx.circuitCenter;
    pilot.radius             = ctx.circuitRadius;
}

void Autopilot::update(double dt)
{
    AircraftBase::update(dt);
    if (read<AircraftState>().crashed) return;
    // Its own driver: the sticks on itself, the lever directly
    const FlightModel m             = flightModelOf(world(), entity());
    write<AircraftControls>()       = pilot.controls(m);
    write<AircraftState>().throttle = pilot.throttle(m);
}

void flightPeerJoined(World& host, uint8_t peer, bool demo)
{
    const Driven d = spawnDriven<Aircraft>(host, peer, peer);
    if (demo) host.add<DemoPilot>(d.driver);
}

WorldSetup flightWorldSetup(std::shared_ptr<const FlightContext> context, const std::string& dataDirectory)
{
    WorldSetup s;
    // The prototype's 32 km space (S20): its seed is the context's (the context evaluates the same surface)
    s.seed                  = context->seed;
    s.config.spaceTemplates = {prototypeSpaceTemplate(context->baseSpacing)};
    s.generators            = registerPrototypeGenerator;
    s.components            = [](Registry& r)
    {
        r.registerComponent<AircraftControls>(Replicate::OwnerAuthoritative);
        r.registerComponent<AircraftState>(Replicate::Host);
        registerDrivingComponents(r);
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Gameplay, "flightDemoPilot", flightDemoPilotSystem, RunOn::Client);
        l.add(Phase::Gameplay, "flightModel", flightModelSystem);
        l.add(Phase::Gameplay, "driverPose", driverPoseSystem, RunOn::Both);
    };
    s.classes = [](ClassList& c)
    {
        c.add<Aircraft>();
        c.add<Autopilot>();
    };
    s.actions       = [](ActionList& a) { a.add(uint32_t(FlightAction::Respawn), "respawn"); };
    s.dataTypes     = [](DataTypeList& d) { d.add<AircraftParameters>(); };
    s.dataDirectory = dataDirectory;
    s.renderAssets  = [](World& w)
    {
        const Vulkan::MaterialId flat = w.registerMaterial("flight/flat", demoMaterial());
        registerDemoMesh(w, kAirplaneMesh, []() { return MeshBuilder::airplane(); }, flat);
    };
    s.setContext(std::move(context));
    return s;
}

// ---- scenarios (--scenarios; docs/engine_design.md §17.2) ----------------------------------------------------------

void flightScenarios(ScenarioList& list)
{
    // The behavior before the port to the engine (S13b): the autopilot's positions after 600 and 1800 steps
    list.add("autopilot_flies_the_circuit",
             [](ScenarioWorld& w)
             {
                 const Entity plane = w.spawn<Autopilot>(Vec3::Zero());
                 w.trace<AircraftState>(plane);
                 auto near = [&](const Vec3& expected, const char* when)
                 {
                     const Vec3 p = w.world().read<Transform>(plane).position();
                     w.check((p - expected).norm() < 1.0, std::string(when) + ": at " + std::to_string(p(0)) + " " +
                                                              std::to_string(p(1)) + " " + std::to_string(p(2)));
                 };
                 w.step(600);
                 near(Vec3(6506.433, 402.839, 4752.384), "step 600");
                 w.step(1200);
                 near(Vec3(5858.335, 362.351, 5875.344), "step 1800");
                 const double altitude = flightModelOf(w.world(), plane).altitudeAboveGround();
                 w.check(std::abs(altitude - 300) < 30, "holds 300 m above the ground: " + std::to_string(altitude));
                 w.check(!w.world().read<AircraftState>(plane).crashed, "did not crash");
             });
    // S16b: a driver of the default peer (a client's in the host + client variants): the lever on the host-owned plane,
    // the sticks written on the driver's owner; stepUntil (the sticks arrive up to 8 steps late)
    list.add("climbs_with_full_throttle",
             [](ScenarioWorld& w)
             {
                 const Driven d                               = spawnDriven<Aircraft>(w.world(), w.defaultPeer(), 0);
                 const Entity plane                           = d.vehicle;
                 const double y0                              = w.world().read<Transform>(plane).position().y();
                 w.world().get<AircraftState>(plane).throttle = 1.0;
                 w.edit(d.driver, [](World& owner, Entity e) { owner.get<AircraftControls>(e).pitch = 0.25; });
                 w.trace<AircraftState>(plane);
                 w.check(
                     w.stepUntil([&] { return w.world().read<Transform>(plane).position().y() > y0 + 60; }, 60 * 10),
                     "climbed " + std::to_string(w.world().read<Transform>(plane).position().y() - y0) + " m");
                 w.check(!w.world().read<AircraftState>(plane).crashed, "did not crash");
             });
    list.add(
        "crashes_into_terrain",
        [](ScenarioWorld& w)
        {
            const Driven d                               = spawnDriven<Aircraft>(w.world(), w.defaultPeer(), 0);
            const Entity plane                           = d.vehicle;
            w.world().get<AircraftState>(plane).throttle = 1.0;
            w.edit(d.driver, [](World& owner, Entity e) { owner.get<AircraftControls>(e).pitch = -1.0; });
            w.trace<AircraftState>(plane);
            w.check(w.stepUntil([&] { return w.world().read<AircraftState>(plane).crashed; }, 60 * 30),
                    "crashes when diving");
            // The crash lasts 2 s, then the plane is back on the circuit (host-side)
            w.stepSeconds(2.5);
            w.check(!w.world().read<AircraftState>(plane).crashed, "respawned");
            const double altitude = flightModelOf(w.world(), plane).altitudeAboveGround();
            w.check(altitude > 250, "300 m above the ground after the respawn: " + std::to_string(altitude));
            // The respawn action of the driver reaches its plane (driver -> Driver.vehicle): level sticks, fly
            // away from the respawn point, then the action brings it back
            w.edit(d.driver, [](World& owner, Entity e) { owner.get<AircraftControls>(e) = AircraftControls(); });
            const FlightContext& ctx = w.world().context<FlightContext>();
            Vec3 spawnPoint          = ctx.circuitCenter + Vec3(ctx.circuitRadius, 0, 0);
            spawnPoint(1)            = ctx.ground(spawnPoint(0), spawnPoint(2)) + 300;
            w.check(w.stepUntil([&] { return (w.world().read<Transform>(plane).position() - spawnPoint).norm() > 100; },
                                60 * 5),
                    "flies away from the respawn point");
            w.input(d.driver, uint32_t(FlightAction::Respawn));
            w.check(
                w.stepUntil([&] { return (w.world().read<Transform>(plane).position() - spawnPoint).norm() < 5; }, 60),
                "the driver's respawn action respawns its plane");
        });
}

}  // namespace Game
}  // namespace Saiga
