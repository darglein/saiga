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
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/gameplay/Systems.h"
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
    const PlatformerContext& context = w.context<PlatformerContext>();
    w.ownedBy<Transform, CharacterMotion, const PlayerInput>(w.localPeer())
        .each(
            [&](Entity, Transform& t, CharacterMotion& m, const PlayerInput& in)
            {
                CharacterController c;
                c.params = w.data(m.model);
                c.level  = &context.levelOf(t.space());  // S21: the level of the hero's space
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

// Host (S21, game policy): an InteractEvent of the hero with the nearest door (SpaceLink) within 1.5 m in its space
void interactWithNearestDoor(World& w, Entity hero)
{
    const Transform& h = w.read<Transform>(hero);
    Entity best;
    double bestDistance = 1.5;
    w.viewAll<const SpaceLink, const Transform>().each(
        [&](Entity door, const SpaceLink&, const Transform& t)
        {
            if (t.space() != h.space()) return;
            const double d = Vec3(t.position()(0) - h.position()(0), 0, t.position()(2) - h.position()(2)).norm();
            if (d < bestDistance || (d == bestDistance && best.valid() && door.index < best.index))
            {
                best         = door;
                bestDistance = d;
            }
        });
    if (best.valid()) w.events.emit(InteractEvent{best, hero, uint32_t(PlatformerAction::Interact)});
}

// The demo runner's input for a character (DemoRunner::update and the owner-side pilot): PlayerInput.move, the held
// jump bit, and a press through jumpRequest. S21: the route of RunState::route; the runner walks to route[waypoint]
// while the hero is in that waypoint's space (the consecutive waypoints of that space are the runner's list, so the
// outdoor route runs exactly as before); at a door within 1 m it stands and queues the interact action every 30 steps
// (the host drops repeats: the first request wins); after a transition the waypoint advances to the next one in the
// hero's new space
void runnerInput(World& w, Entity e, int32_t& waypoint)
{
    const PlatformerContext& context    = w.context<PlatformerContext>();
    const int32_t routeIndex            = w.has<RunState>(e) ? w.read<RunState>(e).route : 0;
    const std::vector<DemoWaypoint>& rt = context.route(routeIndex);
    const SpaceId space                 = w.read<Transform>(e).space();
    PlayerInput& input                  = w.get<PlayerInput>(e);
    CharacterMotion& motion             = w.get<CharacterMotion>(e);
    auto stand                          = [&]()
    {
        input.move         = Vec3::Zero();
        input.heldButtons  = 0;
        motion.jumpRequest = false;
    };
    if (waypoint >= 0 && size_t(waypoint) < rt.size() && rt[size_t(waypoint)].space != space)
    {
        // Through a door: the next waypoint of the new space
        if (size_t(waypoint) + 1 < rt.size() && rt[size_t(waypoint) + 1].space == space)
            waypoint++;
        else
            return stand();
    }
    if (waypoint < 0 || size_t(waypoint) >= rt.size()) return stand();
    const DemoWaypoint& target = rt[size_t(waypoint)];
    if (target.door)
    {
        const Vec3 p = w.read<Transform>(e).position();
        Vec3 d(target.position(0) - p(0), 0, target.position(2) - p(2));
        const double dist = d.norm();
        stand();
        if (dist < 1.0)
        {
            // At the door: the interact action (repeated until the transition happens); the host's demo runner is no
            // player: it uses the door directly (one step less)
            if (w.stepIndex() % 30 == 0)
            {
                if (w.has<Player>(e))
                    w.queueAction(PlayerAction{e, uint32_t(PlatformerAction::Interact), Vec3::Zero()});
                else
                    interactWithNearestDoor(w, e);
            }
            return;
        }
        input.move = d / dist * std::max(std::min(dist / 1.5, 1.0), 0.3);
        return;
    }
    // The consecutive waypoints of the hero's space up to the next door: the runner's list
    Level segment;
    size_t end = size_t(waypoint);
    while (end < rt.size() && !rt[end].door && rt[end].space == space) segment.waypoints.push_back(rt[end++].position);
    const Level& level = context.levelOf(space);
    PlatformerDemo runner;
    runner.next             = 0;
    const CharacterInput ci = runner.input(characterOf(w, e), level, segment.waypoints);
    waypoint += int32_t(runner.next);
    PlayerInput& in = input;
    in.move         = ci.move;
    // The runner holds the jump bit (full jump height) and presses only through a request (never through the bit)
    in.heldButtons     = ci.jumpHeld ? (1u << int(PlatformerButton::Jump)) : 0u;
    CharacterMotion& m = motion;
    m.jumpRequest      = ci.jumpPressed;
    m.jumpWasHeld      = m.jumpWasHeld || ci.jumpHeld;
}

// Phase Gameplay (host; S21): the interact action of a hero becomes an InteractEvent with the nearest door
// (interactWithNearestDoor; the engine's spaceLinkSystem turns it into the transition)
void platformerInteractSystem(World& w, double)
{
    for (const ActionEvent& a : w.events.read<ActionEvent>())
        if (a.action == uint32_t(PlatformerAction::Interact) && w.alive(a.player)) interactWithNearestDoor(w, a.player);
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
// jump, only a fresh press does; the demo pilot starts at the first waypoint again. Regression (PR review, S21): a
// host-owned hero with DemoPilot kept the pilot's waypoint -1 after its restart and stood at the start
void resetMotion(World& w, Entity e)
{
    CharacterMotion& m = w.get<CharacterMotion>(e);
    const auto model   = m.model;
    m                  = CharacterMotion();
    m.model            = model;
    m.jumpWasHeld      = w.has<PlayerInput>(e) && heldBit(w.read<PlayerInput>(e).heldButtons, PlatformerButton::Jump);
    if (w.has<DemoPilot>(e)) w.get<DemoPilot>(e).waypoint = 0;
}

// RestartRun on the hero's owner (phase 1, before the ownerTeleport's OwnerCommand of the same step)
void onRestartRun(World& w, uint8_t, const RestartRun& m)
{
    if (w.isHost()) return;
    const Entity e = w.replication().entityOfNetId(m.hero);
    if (!e.valid() || !w.alive(e) || w.ownerOf(e) != w.localPeer() || !w.has<CharacterMotion>(e)) return;
    resetMotion(w, e);
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
    c.level  = &w.context<PlatformerContext>().levelOf(w.read<Transform>(e).space());
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
    World& w            = world();
    const Level& level  = w.context<PlatformerContext>().outdoor();
    const int32_t slot  = read<RunState>().slot;
    const int32_t route = read<RunState>().route;
    RunState& run       = write<RunState>();
    run                 = RunState();
    run.slot            = slot;
    run.route           = route;
    // PlayerInput stays: the device (or the demo runner) owns it; a held key must not look like a new press
    const uint8_t owner = w.ownerOf(entity());
    if (owner != 0 && w.replication().spawned(entity()))
        // A client's hero (R-NET-2): its owner resets the motion; RestartRun goes before the ownerTeleport's command
        w.send(owner, RestartRun{w.replication().netIdOf(entity())});
    else
        resetMotion(w, entity());
    // S21: back to the outdoor level also from the cellar (for a client's hero an OwnerCommand with space 0)
    w.moveToSpace(entity(), 0, startOf(level, slot), yawRotation(0));
}

// The checks see the result of the previous step's controller (phase Gameplay runs after this one); the run time is
// counted before this step's move, so the goal time equals the time of the step that reached it
void HeroBase::update(double dt)
{
    const Level& level = world().context<PlatformerContext>().levelOf(transform().space());
    bool restartNow    = false;
    for (const ActionEvent& a : world().events.read<ActionEvent>())
        if (a.player == entity() && a.action == uint32_t(PlatformerAction::Restart)) restartNow = true;
    if (restartNow || transform().position().y() < level.killHeight)
    {
        restart();
        return;
    }
    RunState& run = write<RunState>();
    if (run.goalTime < 0 && transform().space() == 0)
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

Entity spawnHero(World& host, uint8_t peer, int slot, PlatformerRoute route)
{
    Player p;
    p.peer        = peer;
    p.characterId = uint64_t(peer) + 1;
    RunState run;
    run.slot       = slot;
    run.route      = int32_t(route);
    const Entity e = host.spawn<PlatformerHero>(Vec3::Zero(), p, PlayerInput{}, run);
    if (peer != 0) host.setOwner(e, peer);
    return e;
}

Entity spawnDemoRunner(World& host, PlatformerRoute route)
{
    RunState run;
    run.route = int32_t(route);
    return host.spawn<DemoRunner>(Vec3::Zero(), run);
}

void platformerPeerJoined(World& host, uint8_t peer, bool demo, PlatformerRoute route)
{
    const Entity e = spawnHero(host, peer, peer, route);
    if (demo) host.add<DemoPilot>(e);
}

std::string doorMeshName()
{
    return "platformer/door";
}

void spawnDoors(World& host)
{
    auto door = [&](SpaceId space, const Vec3& p, SpaceId target, const Vec3& arrival)
    {
        const Entity e = host.create();
        host.moveToSpace(e, space, p, Quat::Identity());
        SpaceLink link;
        link.target  = target;
        link.arrival = arrival;
        host.add<SpaceLink>(e, link);
        host.add<RenderInstance>(e, RenderInstance(doorMeshName()));
    };
    door(0, CellarDoors::entrance(), 1, CellarDoors::cellarArrival());
    door(1, CellarDoors::exit(), 0, CellarDoors::entranceArrival());
}

PlatformerContext PlatformerContext::demo()
{
    PlatformerContext c;
    c.levels = {Level::demoLevel(), Level::cellarLevel()};
    c.routes = {outdoorRoute(c.levels[0]), cellarRoute(c.levels[0])};
    return c;
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
    // S21: RunState version 2 adds the route (the outdoor one before)
    s.migrations = [](MigrationList& m) { m.add<RunState>(1, [](FieldMap& f) { f.set("route", 0); }); };
    // S21: the outdoor level (the flat default template) and the enclosed cellar (its site origin (640, 0, 640))
    SpaceTemplate cellar    = SpaceTemplate::enclosed("cellar", CellarDoors::kCellarChunks);
    cellar.background       = Vec3(0.012, 0.010, 0.009);
    s.config.spaceTemplates = {SpaceTemplate(), cellar};
    s.systems               = [](SystemList& l)
    {
        // The pilot runs on the hero's owner: a client's (--coop --demo), or a host-owned hero with DemoPilot (the
        // single-world scenario variants)
        l.add(Phase::Behaviour, "platformerDemoPilot", platformerDemoPilotSystem, RunOn::Both);
        l.add(Phase::Gameplay, "characterController", characterControllerSystem, RunOn::Both);
        // S21: the doors (the game's interact rule, then the engine's entrance system; one step apart, §12.4)
        l.add(Phase::Gameplay, "platformerInteract", platformerInteractSystem, RunOn::Host);
        l.add(Phase::Gameplay, "space_link", spaceLinkSystem, RunOn::Host);
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
        a.add(uint32_t(PlatformerAction::Interact), "interact");
        a.addHeld(int(PlatformerButton::Jump), "jump");
        a.addHeld(int(PlatformerButton::Walk), "walk");
    };
    s.dataTypes    = [](DataTypeList& d) { d.add<CharacterParameters>(); };
    s.renderAssets = [](World& w)
    {
        const Vulkan::MaterialId flat = w.registerMaterial("platformer/flat", demoMaterial());
        for (int j = 0; j < JointCount; ++j)
            registerDemoMesh(w, heroPartMeshName(j), [j]() { return heroPartMesh(j); }, flat, 500);
        // S21: a door with its frame (drawn by the extract in the door's space)
        registerDemoMesh(
            w, doorMeshName(),
            []()
            {
                MeshData m = MeshBuilder::box(Vec3(0.6, 1.1, 0.06), Vec3(0.38, 0.24, 0.14), Vec3(0, 1.1, 0));
                m.append(MeshBuilder::box(Vec3(0.08, 1.2, 0.1), Vec3(0.3, 0.3, 0.32), Vec3(-0.68, 1.2, 0)));
                m.append(MeshBuilder::box(Vec3(0.08, 1.2, 0.1), Vec3(0.3, 0.3, 0.32), Vec3(0.68, 1.2, 0)));
                m.append(MeshBuilder::box(Vec3(0.76, 0.08, 0.1), Vec3(0.3, 0.3, 0.32), Vec3(0, 2.4, 0)));
                m.append(MeshBuilder::box(Vec3(0.05, 0.05, 0.05), Vec3(0.9, 0.75, 0.3), Vec3(0.4, 1.05, 0.1)));
                return m;
            },
            flat, 500);
    };
    s.setContext(std::move(context));
    return s;
}

// ---- scenarios -----------------------------------------------------------------------------------------------------

namespace
{
// The cellar scenarios' record of a hero's stay in the cellar (space 1, on the host): the first step in it and the
// steps spent there
struct CellarStay
{
    int64_t entry = -1;
    int steps     = 0;
    bool left     = false;
    void observe(ScenarioWorld& w, Entity hero)
    {
        const bool inside = w.world().read<Transform>(hero).space() == 1;
        if (inside && entry < 0) entry = int64_t(w.steps());
        if (inside) steps++;
        if (!inside && entry >= 0) left = true;
    }
};
}  // namespace

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
    // S21: a hero of the default peer uses the entrance (the interact action at the door), walks through the cellar
    // around the platforms (steered on the owner from its own pose) and leaves through the exit
    list.add("cellar_round_trip",
             [](ScenarioWorld& w)
             {
                 spawnDoors(w.world());
                 const Entity hero = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), w.defaultPeer());
                 // Before its spawn was sent: a plain move also for a client's hero
                 w.world().moveToSpace(hero, 0, CellarDoors::entrance() + Vec3(0.6, 0, 0), yawRotation(0));
                 w.step(10);
                 auto space = [&]() { return w.world().read<Transform>(hero).space(); };
                 w.input(hero, uint32_t(PlatformerAction::Interact));
                 w.check(w.stepUntil([&] { return space() == 1; }, 40), "in the cellar");
                 w.check((w.world().read<Transform>(hero).position() - CellarDoors::cellarArrival()).norm() < 1e-3,
                         "at the cellar's arrival");
                 if (w.networked())
                 {
                     auto ownerSpace = [&]()
                     {
                         const Entity c = w.onClient(w.defaultPeer(), hero);
                         return c.valid() ? int(w.client(w.defaultPeer()).read<Transform>(c).space()) : -1;
                     };
                     w.check(w.stepUntil([&] { return ownerSpace() == 1; }, 20), "the owner's hero in the cellar");
                 }
                 // Around the platforms on the left to the exit
                 const Vec3 o                   = CellarDoors::origin();
                 const std::vector<Vec3> points = {o + Vec3(-3.5, 0, 0), o + Vec3(-3.5, 0, -19), CellarDoors::exit()};
                 auto steer                     = [&]()
                 {
                     w.edit(hero,
                            [points](World& owner, Entity e)
                            {
                                const Vec3 p = owner.read<Transform>(e).position();
                                size_t k     = 0;
                                if (p.z() < points[0].z() + 0.5) k = 1;
                                if (p.z() < points[1].z() + 0.5) k = 2;
                                Vec3 d(points[k].x() - p.x(), 0, points[k].z() - p.z());
                                PlayerInput& in = owner.get<PlayerInput>(e);
                                in.move         = d.norm() > 0.3 ? Vec3(d.normalized()) : Vec3::Zero();
                                in.heldButtons  = 0;
                            });
                     const Vec3 p = w.world().read<Transform>(hero).position();
                     return Vec3(p.x() - CellarDoors::exit().x(), 0, p.z() - CellarDoors::exit().z()).norm() < 1.0;
                 };
                 w.check(w.stepUntil(steer, 60 * 10), "walked to the exit");
                 w.edit(hero, [](World& owner, Entity e) { owner.get<PlayerInput>(e).move = Vec3::Zero(); });
                 w.step(5);
                 w.input(hero, uint32_t(PlatformerAction::Interact));
                 w.check(w.stepUntil([&] { return space() == 0; }, 40), "back outside");
                 // The controller of the arrival step ran after the move (phase 4 after phase 1): within its settling
                 const Vec3 back = w.world().read<Transform>(hero).position();
                 w.check((back - CellarDoors::entranceArrival()).norm() < 0.05,
                         "next to the entrance: " + std::to_string((back - CellarDoors::entranceArrival()).norm()));
             });
    // S21: the host's demo runner on the cellar route: through the cellar (at least 240 steps there: the pinned
    // pre-simulation of --demo --cellar), then the outdoor level to the goal
    list.add("demo_runner_visits_the_cellar",
             [](ScenarioWorld& w)
             {
                 spawnDoors(w.world());
                 const Entity hero = spawnDemoRunner(w.world(), PlatformerRoute::Cellar);
                 w.trace<RunState>(hero);
                 CellarStay stay;
                 w.check(w.stepUntil(
                             [&]
                             {
                                 stay.observe(w, hero);
                                 return w.world().read<RunState>(hero).goalTime >= 0;
                             },
                             60 * 40),
                         "reaches the goal");
                 w.check(stay.left, "left the cellar");
                 w.check(stay.steps >= 240, "stays at least 240 steps in the cellar: " + std::to_string(stay.steps));
                 SAIGA_LOG("platformer", "demo_runner_visits_the_cellar: entry step %lld, stay %d steps, goal %.2f s",
                           (long long)stay.entry, stay.steps, w.world().read<RunState>(hero).goalTime);
             });
    // S21: a hero of the default peer with the demo pilot on the cellar route (on the client in the network variants):
    // in the cellar and back outside within a bounded step count
    list.add("pilot_visits_the_cellar",
             [](ScenarioWorld& w)
             {
                 spawnDoors(w.world());
                 RunState run;
                 run.route         = int32_t(PlatformerRoute::Cellar);
                 const Entity hero = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), w.defaultPeer(), run);
                 w.world().add<DemoPilot>(hero);
                 CellarStay stay;
                 w.check(w.stepUntil(
                             [&]
                             {
                                 stay.observe(w, hero);
                                 return stay.left;
                             },
                             60 * 30),
                         "in the cellar and back outside");
                 w.check(stay.steps >= 240, "stays at least 240 steps in the cellar: " + std::to_string(stay.steps));
                 SAIGA_LOG("platformer", "pilot_visits_the_cellar: entry step %lld, stay %d steps",
                           (long long)stay.entry, stay.steps);
             });
    // The demo pilot's restart after the goal (the restart action) and the next run (host-owned in the single-world
    // variants, the client's in the network ones). Regression (PR review, S21): a host-owned pilot stood at the start
    // after its restart (the waypoint stayed -1)
    list.add("pilot_runs_again_after_the_restart",
             [](ScenarioWorld& w)
             {
                 const Entity hero = w.spawnPlayer<PlatformerHero>(Vec3::Zero(), w.defaultPeer(), RunState{});
                 w.world().add<DemoPilot>(hero);
                 auto run = [&]() -> const RunState& { return w.world().read<RunState>(hero); };
                 w.check(w.stepUntil([&] { return run().goalTime >= 0; }, 60 * 40), "reaches the goal");
                 w.check(w.stepUntil([&] { return run().goalTime < 0; }, 60 * 6), "restarted after the goal");
                 const Vec3 start = w.world().read<Transform>(hero).position();
                 w.check(w.stepUntil([&] { return (w.world().read<Transform>(hero).position() - start).norm() > 3; },
                                     60 * 5),
                         "runs again from the start");
             });
}

}  // namespace Game
}  // namespace Saiga
