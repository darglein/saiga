/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The flight simulator demo in a World: components, the flight model system, Aircraft and Autopilot, scenarios.

#include "games/flight/FlightWorld.h"

#include "saiga/engine/components/Player.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"

#include <chrono>
#include <iostream>

namespace Saiga
{
namespace Game
{
using namespace Engine;

std::shared_ptr<const FlightContext> FlightContext::create()
{
    auto c        = std::make_shared<FlightContext>();
    const auto t0 = std::chrono::steady_clock::now();
    c->heightField.generate(Vulkan::HeightField::Parameters());
    const double size = c->heightField.worldSize();
    c->circuitCenter  = Vec3(size / 2, 0, size / 2);
    c->circuitRadius  = size * 0.3;
    std::cout << "Flight: height field " << c->heightField.size() << "^2 generated in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
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
// Phase Gameplay: the flight model of every airplane (after the pilots wrote the controls in phase Behaviour)
void flightModelSystem(World& w, double dt)
{
    w.view<Transform, AircraftState, AircraftControls>().each(
        [&](Entity e, Transform& t, AircraftState& s, AircraftControls& c)
        {
            if (s.crashed)
            {
                s.crashTimer += dt;
                return;
            }
            c.throttle    = clampd(c.throttle + c.throttleChange * 0.5 * dt, 0, 1);
            FlightModel m = flightModelOf(w, e);
            m.step(c, dt);
            t.setPosition(m.state.position);
            t.setRotation(m.state.orientation);
            s.velocity  = m.state.velocity;
            s.bodyRates = m.state.bodyRates;
            s.onGround  = m.state.onGround;
            s.crashed   = m.state.crashed;
        });
}
}  // namespace

void AircraftBase::onSpawn()
{
    if (!has<AircraftState>()) add<AircraftState>();
    if (!has<AircraftControls>()) add<AircraftControls>(AircraftControls{0.6, 0, 0, 0, 0});
    respawn();
}

void AircraftBase::update(double)
{
    bool requested = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (a.player == entity() && a.action == uint32_t(FlightAction::Respawn)) requested = true;
    if (requested || (read<AircraftState>().crashed && read<AircraftState>().crashTimer > 2.0)) respawn();
}

void AircraftBase::respawn()
{
    const FlightContext& ctx = world().context<FlightContext>();
    Vec3 p                   = ctx.circuitCenter + Vec3(ctx.circuitRadius, 0, 0);
    p(1)                     = ctx.ground(p(0), p(2)) + 300;
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
    write<AircraftControls>() = pilot.controls(flightModelOf(world(), entity()));
}

WorldSetup flightWorldSetup(std::shared_ptr<const FlightContext> context, const std::string& dataDirectory)
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<AircraftControls>();
        r.registerComponent<AircraftState>();
    };
    s.systems = [](SystemList& l) { l.add(Phase::Gameplay, "flightModel", flightModelSystem); };
    s.classes = [](ClassList& c)
    {
        c.add<Aircraft>();
        c.add<Autopilot>();
    };
    s.actions       = [](ActionList& a) { a.add(uint32_t(FlightAction::Respawn), "respawn"); };
    s.dataTypes     = [](DataTypeList& d) { d.add<AircraftParameters>(); };
    s.dataDirectory = dataDirectory;
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
    list.add("climbs_with_full_throttle",
             [](ScenarioWorld& w)
             {
                 const Entity plane                     = w.spawnPlayer<Aircraft>(Vec3::Zero(), 0);
                 const double y0                        = w.world().read<Transform>(plane).position().y();
                 w.world().get<AircraftControls>(plane) = AircraftControls{1.0, 0.25, 0, 0, 0};
                 w.trace<AircraftState>(plane);
                 w.stepSeconds(8);
                 const double y = w.world().read<Transform>(plane).position().y();
                 w.check(y > y0 + 60, "climbed " + std::to_string(y - y0) + " m");
                 w.check(!w.world().read<AircraftState>(plane).crashed, "did not crash");
             });
    list.add("crashes_into_terrain",
             [](ScenarioWorld& w)
             {
                 const Entity plane                     = w.spawnPlayer<Aircraft>(Vec3::Zero(), 0);
                 w.world().get<AircraftControls>(plane) = AircraftControls{1.0, -1.0, 0, 0, 0};
                 w.trace<AircraftState>(plane);
                 w.check(w.stepUntil([&] { return w.world().read<AircraftState>(plane).crashed; }, 60 * 30),
                         "crashes when diving");
                 // The crash lasts 2 s, then the plane is back on the circuit
                 w.stepSeconds(2.5);
                 w.check(!w.world().read<AircraftState>(plane).crashed, "respawned");
                 const double altitude = flightModelOf(w.world(), plane).altitudeAboveGround();
                 w.check(altitude > 250, "300 m above the ground after the respawn: " + std::to_string(altitude));
             });
}

}  // namespace Game
}  // namespace Saiga
