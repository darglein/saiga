/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The platformer demo in a World: components, the character controller system, the hero and the demo runner,
// scenarios.

#include "games/platformer/PlatformerWorld.h"

#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"

namespace Saiga
{
namespace Game
{
using namespace Engine;

namespace
{
bool heldBit(uint32_t bits, PlatformerButton b)
{
    return (bits >> int(b)) & 1u;
}

CharacterState stateOf(const Transform& t, const CharacterMotion& m)
{
    CharacterState s;
    s.position        = t.position();
    s.velocity        = m.velocity;
    s.yaw             = m.yaw;
    s.grounded        = m.grounded;
    s.airTime         = m.airTime;
    s.landingSpeed    = m.landingSpeed;
    s.jumped          = m.jumped;
    s.jumpBufferTimer = m.jumpBufferTimer;
    return s;
}

// Phase Gameplay: the kinematic controller of every character, from its PlayerInput (move in world space, the jump
// and walk bits)
void characterControllerSystem(World& w, double dt)
{
    const Level* level = &w.context<PlatformerContext>().level;
    w.view<Transform, CharacterMotion, const PlayerInput>().each(
        [&](Entity, Transform& t, CharacterMotion& m, const PlayerInput& in)
        {
            CharacterController c;
            c.params = w.data(m.model);
            c.level  = level;
            c.state  = stateOf(t, m);
            CharacterInput ci;
            ci.move        = in.move;
            ci.walk        = heldBit(in.heldButtons, PlatformerButton::Walk);
            ci.jumpHeld    = heldBit(in.heldButtons, PlatformerButton::Jump);
            ci.jumpPressed = (ci.jumpHeld && !m.jumpWasHeld) || m.jumpRequest;
            m.jumpWasHeld  = ci.jumpHeld;
            m.jumpRequest  = false;
            c.step(ci, dt);
            t.setPosition(c.state.position);
            t.setRotation(yawRotation(c.state.yaw));
            m.velocity        = c.state.velocity;
            m.yaw             = c.state.yaw;
            m.grounded        = c.state.grounded;
            m.airTime         = c.state.airTime;
            m.landingSpeed    = c.state.landingSpeed;
            m.jumped          = c.state.jumped;
            m.jumpBufferTimer = c.state.jumpBufferTimer;
        });
}
}  // namespace

CharacterController characterOf(const World& w, Entity e)
{
    CharacterController c;
    c.params = w.data(w.read<CharacterMotion>(e).model);
    c.level  = &w.context<PlatformerContext>().level;
    c.state  = stateOf(w.read<Transform>(e), w.read<CharacterMotion>(e));
    return c;
}

void HeroBase::onSpawn()
{
    if (!has<CharacterMotion>()) add<CharacterMotion>();
    if (!has<RunState>()) add<RunState>();
    if (!has<PlayerInput>()) add<PlayerInput>();
    restart();
}

void HeroBase::restart()
{
    const Level& level = world().context<PlatformerContext>().level;
    CharacterMotion& m = write<CharacterMotion>();
    const auto model   = m.model;
    m                  = CharacterMotion();
    m.model            = model;
    // A jump key held through the restart does not jump: only a fresh press does
    m.jumpWasHeld     = true;
    write<RunState>() = RunState();
    // PlayerInput stays: the device (or the demo runner) owns it; a held key must not look like a new press
    transformWrite().teleport(level.start, yawRotation(0));
}

// The checks see the result of the previous step's controller (phase Gameplay runs after this one); the run time is
// counted before this step's move, so the goal time equals the time of the step that reached it
void HeroBase::update(double dt)
{
    const Level& level = world().context<PlatformerContext>().level;
    bool restartNow    = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (a.player == entity() && a.action == uint32_t(PlatformerAction::Restart)) restartNow = true;
    if (restartNow || transform().position().y() < level.killHeight)
    {
        restart();
        return;
    }
    RunState& run = write<RunState>();
    if (run.goalTime < 0)
    {
        PlatformerDemo probe;
        if (probe.reachedGoal(characterOf(world(), entity()), level)) run.goalTime = run.runTime;
    }
    if (restartsAfterTheGoal() && run.goalTime >= 0 && run.runTime > run.goalTime + 3)
    {
        restart();
        return;
    }
    run.runTime += dt;
}

void DemoRunner::onSpawn()
{
    HeroBase::onSpawn();
}

void DemoRunner::update(double dt)
{
    HeroBase::update(dt);
    const Level& level = world().context<PlatformerContext>().level;
    PlatformerDemo runner;
    runner.next                = size_t(read<RunState>().waypoint);
    const CharacterInput ci    = runner.input(characterOf(world(), entity()), level);
    write<RunState>().waypoint = int32_t(runner.next);
    PlayerInput& in            = write<PlayerInput>();
    in.move                    = ci.move;
    // The runner holds the jump bit (full jump height) and presses only through a request (never through the bit)
    in.heldButtons     = ci.jumpHeld ? (1u << int(PlatformerButton::Jump)) : 0u;
    CharacterMotion& m = write<CharacterMotion>();
    m.jumpRequest      = ci.jumpPressed;
    m.jumpWasHeld      = m.jumpWasHeld || ci.jumpHeld;
}

WorldSetup platformerWorldSetup(std::shared_ptr<const PlatformerContext> context)
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<CharacterMotion>();
        r.registerComponent<RunState>();
    };
    s.systems = [](SystemList& l) { l.add(Phase::Gameplay, "characterController", characterControllerSystem); };
    s.classes = [](ClassList& c)
    {
        c.add<PlatformerHero>();
        c.add<DemoRunner>();
    };
    s.actions = [](ActionList& a)
    {
        a.add(uint32_t(PlatformerAction::Restart), "restart");
        a.addHeld(int(PlatformerButton::Jump), "jump");
        a.addHeld(int(PlatformerButton::Walk), "walk");
    };
    s.dataTypes = [](DataTypeList& d) { d.add<CharacterParameters>(); };
    s.setContext(std::move(context));
    return s;
}

// ---- scenarios -----------------------------------------------------------------------------------------------------

void platformerScenarios(ScenarioList& list)
{
    // The behavior before the port to the engine (S13b): the demo runner reaches the goal after 7.25 s
    list.add("demo_reaches_goal",
             [](ScenarioWorld& w)
             {
                 const Entity hero = w.spawn<DemoRunner>(Vec3::Zero());
                 w.trace<RunState>(hero);
                 w.check(w.stepUntil([&] { return w.world().read<RunState>(hero).goalTime >= 0; }, 60 * 20),
                         "reaches the goal");
                 const double t = w.world().read<RunState>(hero).goalTime;
                 w.check(std::abs(t - 7.25) < 0.1, "in 7.25 s: " + std::to_string(t));
             });
    list.add("jumps_onto_platform",
             [](ScenarioWorld& w)
             {
                 // From the start area towards the first platform (top 0.8 m, z -11 .. -8) over the gap at z -6 .. -8
                 const Entity hero = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), 0);
                 w.world().get<Transform>(hero).teleport(Vec3(0, 0, -3), yawRotation(0));
                 w.trace<CharacterMotion>(hero);
                 auto onPlatform = [&]()
                 {
                     const Vec3 p    = w.world().read<Transform>(hero).position();
                     PlayerInput& in = w.world().get<PlayerInput>(hero);
                     in.move         = Vec3(0, 0, -1);
                     in.heldButtons  = p.z() < -5.2 ? (1u << int(PlatformerButton::Jump)) : 0u;
                     return w.world().read<CharacterMotion>(hero).grounded && std::abs(p.y() - 0.8) < 0.01 &&
                            p.z() < -8;
                 };
                 w.check(w.stepUntil(onPlatform, 60 * 3), "lands on the platform");
                 w.check(w.world().read<RunState>(hero).goalTime < 0, "not at the goal");
             });
    // Regression: the jump key held through a restart (here: falling below the level) jumped right after it
    list.add("held_jump_after_a_restart",
             [](ScenarioWorld& w)
             {
                 const Entity hero                            = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), 0);
                 const uint32_t jump                          = 1u << int(PlatformerButton::Jump);
                 w.world().get<PlayerInput>(hero).heldButtons = jump;
                 w.step(10);
                 w.check(w.world().read<Transform>(hero).position().y() < 0.01,
                         "a held key at the spawn does not jump");
                 w.world().get<Transform>(hero).teleport(Vec3(0, -20, 2), yawRotation(0));
                 double highest = -1e9;
                 for (int i = 0; i < 40; ++i)
                 {
                     w.world().get<PlayerInput>(hero).heldButtons = jump;
                     w.step();
                     highest = std::max(highest, w.world().read<Transform>(hero).position().y());
                 }
                 w.check(highest < 0.01, "restarted without a jump (highest " + std::to_string(highest) + " m)");
                 w.world().get<PlayerInput>(hero).heldButtons = 0;
                 w.step();
                 w.world().get<PlayerInput>(hero).heldButtons = jump;
                 w.step(10);
                 w.check(w.world().read<Transform>(hero).position().y() > 0.5, "a fresh press jumps");
             });
}

}  // namespace Game
}  // namespace Saiga
