/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The platformer demo in a World: components, the character controller system, the hero and the demo runner,
// scenarios.

#include "games/platformer/PlatformerWorld.h"

#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/net/MessageList.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/tools/ScenarioList.h"
#include "saiga/engine/tools/ScenarioWorld.h"

#include "games/common/RenderAssets.h"
#include "games/platformer/CharacterAnimation.h"

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

// Phase Gameplay, RunOn::Both: the kinematic controller of every character the local peer owns (R-NET-2: the host's
// own and the demo runner on the host, a client's hero on the client), from its PlayerInput (move in world space, the
// jump and walk bits)
void characterControllerSystem(World& w, double dt)
{
    const Level* level = &w.context<PlatformerContext>().level;
    w.ownedBy<Transform, CharacterMotion, const PlayerInput>(w.localPeer())
        .each(
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

// The demo runner's input for a character (DemoRunner::update and the owner-side pilot): PlayerInput.move, the held
// jump bit, and a press through jumpRequest
void runnerInput(World& w, Entity e, int32_t& waypoint)
{
    const Level& level = w.context<PlatformerContext>().level;
    PlatformerDemo runner;
    runner.next             = size_t(waypoint);
    const CharacterInput ci = runner.input(characterOf(w, e), level);
    waypoint                = int32_t(runner.next);
    PlayerInput& in         = w.get<PlayerInput>(e);
    in.move                 = ci.move;
    // The runner holds the jump bit (full jump height) and presses only through a request (never through the bit)
    in.heldButtons     = ci.jumpHeld ? (1u << int(PlatformerButton::Jump)) : 0u;
    CharacterMotion& m = w.get<CharacterMotion>(e);
    m.jumpRequest      = ci.jumpPressed;
    m.jumpWasHeld      = m.jumpWasHeld || ci.jumpHeld;
}

// Phase Behaviour on clients (--coop --demo): the demo runner runs the owned heroes with DemoPilot (before the
// controller, as DemoRunner::update on the host); the waypoint restarts with RestartRun. The demo's restart after the
// goal is the pilot's decision on the owner (a player who took over is not sent back): it queues the restart action
// once (waypoint -1 until RestartRun arrives; the hero stands meanwhile)
void platformerDemoPilotSystem(World& w, double)
{
    std::vector<Entity> heroes;
    w.ownedBy<const DemoPilot, const CharacterMotion, const PlayerInput>(w.localPeer())
        .each(
            [&](Entity e, const DemoPilot& p, const CharacterMotion&, const PlayerInput&)
            {
                if (!p.suspended) heroes.push_back(e);
            });
    for (Entity e : heroes)
    {
        DemoPilot& pilot = w.get<DemoPilot>(e);
        if (pilot.waypoint < 0)
        {
            w.get<PlayerInput>(e).move        = Vec3::Zero();
            w.get<PlayerInput>(e).heldButtons = 0;
            continue;
        }
        const RunState* run = w.has<RunState>(e) ? &w.read<RunState>(e) : nullptr;
        if (run && w.has<Player>(e) && run->goalTime >= 0 && run->runTime > run->goalTime + 3)
        {
            pilot.waypoint = -1;
            w.queueAction(PlayerAction{e, uint32_t(PlatformerAction::Restart), Vec3::Zero()});
            continue;
        }
        runnerInput(w, e, pilot.waypoint);
    }
}

// The owner part of a restart: the controller state (the model stays); a jump key held through the restart does not
// jump, only a fresh press does
void resetMotion(World& w, Entity e)
{
    CharacterMotion& m = w.get<CharacterMotion>(e);
    const auto model   = m.model;
    m                  = CharacterMotion();
    m.model            = model;
    m.jumpWasHeld      = w.has<PlayerInput>(e) && heldBit(w.read<PlayerInput>(e).heldButtons, PlatformerButton::Jump);
}

// RestartRun on the hero's owner (phase 1, before the ownerTeleport's OwnerCommand of the same step)
void onRestartRun(World& w, uint8_t, const RestartRun& m)
{
    if (w.isHost()) return;
    const Entity e = w.replication().entityOfNetId(m.hero);
    if (!e.valid() || !w.alive(e) || w.ownerOf(e) != w.localPeer() || !w.has<CharacterMotion>(e)) return;
    resetMotion(w, e);
    if (w.has<DemoPilot>(e)) w.get<DemoPilot>(e).waypoint = 0;
}

Vec3 startOf(const Level& level, int slot)
{
    // Next to the start, three per row (the start island is 12 m wide; slot 0 is the level's start)
    slot = std::max(0, slot);
    return level.start + Vec3(-1.5 * (slot % 3), 0, 1.2 * ((slot / 3) % 3));
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
    // Tracked by the extract, not drawn (interpolatedPose; the body parts are the renderer's cosmetic instances until
    // the skinned meshes of S34)
    if (!has<RenderInstance>()) add<RenderInstance>();
    restart();
}

void HeroBase::restart()
{
    World& w           = world();
    const Level& level = w.context<PlatformerContext>().level;
    const int32_t slot = read<RunState>().slot;
    RunState& run      = write<RunState>();
    run                = RunState();
    run.slot           = slot;
    // PlayerInput stays: the device (or the demo runner) owns it; a held key must not look like a new press
    const uint8_t owner = w.ownerOf(entity());
    if (owner != 0 && w.replication().spawned(entity()))
        // A client's hero (R-NET-2): its owner resets the motion; RestartRun goes before the ownerTeleport's command
        w.send(owner, RestartRun{w.replication().netIdOf(entity())});
    else
        resetMotion(w, entity());
    w.ownerTeleport(entity(), startOf(level, slot), yawRotation(0));
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
    // The demo restarts after the goal (a co-op client's hero run by the demo runner: its owner's pilot queues the
    // restart action, platformerDemoPilotSystem; local review: the host restarted it also when the player had taken
    // over, which only the owner knows)
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
    // Host-owned: the runner writes the hero's input here
    int32_t waypoint = read<RunState>().waypoint;
    runnerInput(world(), entity(), waypoint);
    write<RunState>().waypoint = waypoint;
}

Entity spawnHero(World& host, uint8_t peer, int slot)
{
    Player p;
    p.peer        = peer;
    p.characterId = uint64_t(peer) + 1;
    RunState run;
    run.slot       = slot;
    const Entity e = host.spawn<PlatformerHero>(Vec3::Zero(), p, PlayerInput{}, run);
    if (peer != 0) host.setOwner(e, peer);
    return e;
}

void platformerPeerJoined(World& host, uint8_t peer, bool demo)
{
    const Entity e = spawnHero(host, peer, peer);
    if (demo) host.add<DemoPilot>(e);
}

std::string heroPartMeshName(int joint)
{
    return "platformer/hero/part" + std::to_string(joint);
}

MeshData heroPartMesh(int joint)
{
    // Body parts: boxes hanging from their joints
    const Vec3 skin(0.95, 0.75, 0.6), shirt(0.25, 0.45, 0.8), pants(0.3, 0.3, 0.35), shoes(0.2, 0.15, 0.12);
    switch (joint)
    {
        case JointPelvis:
            return MeshBuilder::box(Vec3(0.18, 0.1, 0.11), pants, Vec3(0, 0, 0));
        case JointSpine:
            return MeshBuilder::box(Vec3(0.22, 0.25, 0.13), shirt, Vec3(0, 0.27, 0));
        case JointUpperArmL:
        case JointUpperArmR:
            return MeshBuilder::box(Vec3(0.065, 0.15, 0.065), shirt, Vec3(0, -0.14, 0));
        case JointLowerArmL:
        case JointLowerArmR:
            return MeshBuilder::box(Vec3(0.055, 0.15, 0.055), skin, Vec3(0, -0.15, 0));
        case JointThighL:
        case JointThighR:
            return MeshBuilder::box(Vec3(0.085, 0.22, 0.085), pants, Vec3(0, -0.22, 0));
        case JointShinL:
        case JointShinR:
        {
            // Shins with feet
            MeshData shin = MeshBuilder::box(Vec3(0.075, 0.2, 0.075), pants, Vec3(0, -0.2, 0));
            shin.append(MeshBuilder::box(Vec3(0.08, 0.05, 0.13), shoes, Vec3(0, -0.4, -0.05)));
            return shin;
        }
        default:
        {
            // The head with eyes, so the facing is visible
            MeshData head = MeshBuilder::box(Vec3(0.14, 0.15, 0.14), skin, Vec3(0, 0.15, 0));
            head.append(MeshBuilder::box(Vec3(0.025, 0.03, 0.01), Vec3(0.1, 0.1, 0.12), Vec3(-0.06, 0.18, -0.14)));
            head.append(MeshBuilder::box(Vec3(0.025, 0.03, 0.01), Vec3(0.1, 0.1, 0.12), Vec3(0.06, 0.18, -0.14)));
            return head;
        }
    }
}

WorldSetup platformerWorldSetup(std::shared_ptr<const PlatformerContext> context)
{
    WorldSetup s;
    s.components = [](Registry& r)
    {
        r.registerComponent<CharacterMotion>(Replicate::OwnerAuthoritative);
        r.registerComponent<RunState>(Replicate::Host);
        registerDrivingComponents(r);
    };
    s.systems = [](SystemList& l)
    {
        l.add(Phase::Behaviour, "platformerDemoPilot", platformerDemoPilotSystem, RunOn::Client);
        l.add(Phase::Gameplay, "characterController", characterControllerSystem, RunOn::Both);
    };
    s.messages = [](MessageList& m) { m.add<RestartRun>(Reliability::Reliable, onRestartRun); };
    s.classes  = [](ClassList& c)
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
    s.dataTypes    = [](DataTypeList& d) { d.add<CharacterParameters>(); };
    s.renderAssets = [](World& w)
    {
        const Vulkan::MaterialId flat = w.registerMaterial("platformer/flat", demoMaterial());
        for (int j = 0; j < JointCount; ++j)
            registerDemoMesh(w, heroPartMeshName(j), [j]() { return heroPartMesh(j); }, flat, 500);
    };
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
    // S16b: a hero of the default peer (a client's in the host + client variants); the jump is decided inside edit from
    // the owner's own Transform (not the host's extrapolated pose) and the held bit written there
    list.add("jumps_onto_platform",
             [](ScenarioWorld& w)
             {
                 // From the start area towards the first platform (top 0.8 m, z -11 .. -8) over the gap at z -6 .. -8
                 const Entity hero = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), w.defaultPeer());
                 // Before its spawn was sent: a plain teleport also for a client's hero
                 w.world().ownerTeleport(hero, Vec3(0, 0, -3), yawRotation(0));
                 w.trace<CharacterMotion>(hero);
                 auto onPlatform = [&]()
                 {
                     w.edit(hero,
                            [](World& owner, Entity e)
                            {
                                const Vec3 p    = owner.read<Transform>(e).position();
                                PlayerInput& in = owner.get<PlayerInput>(e);
                                in.move         = Vec3(0, 0, -1);
                                in.heldButtons  = p.z() < -5.2 ? (1u << int(PlatformerButton::Jump)) : 0u;
                            });
                     const Vec3 p = w.world().read<Transform>(hero).position();
                     return w.world().read<CharacterMotion>(hero).grounded && std::abs(p.y() - 0.8) < 0.01 &&
                            p.z() < -8;
                 };
                 w.check(w.stepUntil(onPlatform, 60 * 3), "lands on the platform");
                 w.check(w.world().read<RunState>(hero).goalTime < 0, "not at the goal");
             });
    // Regression: the jump key held through a restart (here: falling below the level) jumped right after it. S16b: a
    // client's hero; the fall is an owner-side teleport (edit), the restart RestartRun + ownerTeleport from the host;
    // the key release is held 5 host steps (the asynchronous variant's bound is 2) so the owner sees it; stepUntil for
    // the jumps (the input reaches the host's view of the hero 1 + 2 delay steps later)
    list.add("held_jump_after_a_restart",
             [](ScenarioWorld& w)
             {
                 const Entity hero   = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), w.defaultPeer());
                 const uint32_t jump = 1u << int(PlatformerButton::Jump);
                 auto hold           = [&](uint32_t bits)
                 { w.edit(hero, [bits](World& owner, Entity e) { owner.get<PlayerInput>(e).heldButtons = bits; }); };
                 auto height = [&]() { return w.world().read<Transform>(hero).position().y(); };
                 w.step(5);
                 // Pressed and held: one jump, then standing again with the key still held
                 hold(jump);
                 w.check(w.stepUntil([&] { return height() > 0.5; }, 30), "a press jumps");
                 w.check(w.stepUntil(
                             [&]
                             {
                                 return w.world().read<CharacterMotion>(hero).grounded &&
                                        w.world().read<Transform>(hero).position().y() < 0.01;
                             },
                             120),
                         "lands while the key is held");
                 // Falling below the level restarts at the start; the key is still held
                 w.edit(hero, [](World& owner, Entity e)
                        { owner.get<Transform>(e).teleport(Vec3(0, -20, 2), yawRotation(0)); });
                 double highest = -1e9;
                 for (int i = 0; i < 40; ++i)
                 {
                     w.step();
                     highest = std::max(highest, height());
                 }
                 w.check(highest < 0.01, "restarted without a jump (highest " + std::to_string(highest) + " m)");
                 w.check(std::abs(height()) < 0.01, "back at the start: " + std::to_string(height()) + " m");
                 hold(0);
                 w.step(5);
                 hold(jump);
                 w.check(w.stepUntil([&] { return height() > 0.5; }, 30), "a fresh press jumps");
             });
}

}  // namespace Game
}  // namespace Saiga
