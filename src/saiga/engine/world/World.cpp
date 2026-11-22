/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// World: registration, commands, the step loop, advance and the hash.

#include "saiga/engine/world/World.h"

#include "saiga/engine/components/EngineComponents.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/SpaceLink.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/net/Varint.h"
#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RenderSink.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/TransformSystem.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>

namespace Saiga
{
namespace Engine
{

World::World(const WorldSetup& setup)
    : events(&reg),
      pool(std::max(0, setup.workerThreads - 1)),
      rng(setup.seed),
      contextPtr(setup.context),
      contextType(setup.contextType),
      config(setup.config),
      stepDt(1.0 / setup.config.stepsPerSecond)
{
    SAIGA_ASSERT(setup.workerThreads >= 1, "WorldSetup::workerThreads must be at least 1 (the calling thread)");
    SAIGA_ASSERT(std::isfinite(setup.config.stepsPerSecond) && setup.config.stepsPerSecond > 0 &&
                     setup.config.maxStepsPerAdvance >= 1,
                 "invalid WorldConfig");
    reg.setJobPool(&pool);
    // The spaces (S20): checked first (a template error is a setup error, fatal in every build, D1)
    {
        validateSpaceTemplates(setup.config.spaceTemplates);
        std::vector<SpaceTemplate> templates = setup.config.spaceTemplates;
        if (templates.empty()) templates.push_back(SpaceTemplate());
        std::vector<uint64_t> seeds;
        for (size_t i = 0; i < templates.size(); ++i) seeds.push_back(i == 0 ? setup.seed : hashCombine(setup.seed, i));
        terrainSys = std::make_unique<TerrainSystem>(templates, seeds);
    }
    extractor = std::make_unique<ExtractSystem>(*this);
    // Before any entity exists: every create reaches the grid through the log
    gridConsumer = log.addConsumer();

    // Engine lists first, then the game's (§7.9, §18.2)
    registerEngineComponents(reg);
    if (setup.components) setup.components(reg);
    reg.validateRegistration();

    EventRegistry eventRegistry(events);
    registerEngineEvents(eventRegistry);
    if (setup.events) setup.events(eventRegistry);
    events.validateRegistration();

    registerEngineSystems(systems);
    if (setup.systems) setup.systems(systems);

    if (setup.classes) setup.classes(classes);
    classes.validateRegistration();
    if (setup.migrations) setup.migrations(migrationList);
    if (setup.actions) setup.actions(actionNames);
    // Network messages: the engine's, then the game's (R-REG-1; errors are fatal in every build)
    registerEngineMessages(messages);
    messages.beginGameMessages();
    if (setup.messages) setup.messages(messages);
    net      = setup.transport;
    role     = setup.role;
    selfPeer = setup.localPeer;
    if (net && net->localPeer() != selfPeer)
        SAIGA_EXIT_ERROR("WorldSetup::localPeer " + std::to_string(selfPeer) + " differs from the transport's peer " +
                         std::to_string(net->localPeer()));
    if (role == NetRole::Host && selfPeer != 0) SAIGA_EXIT_ERROR("WorldSetup: the host is peer 0");
    if (role == NetRole::Client && (selfPeer == 0 || !net))
        SAIGA_EXIT_ERROR("WorldSetup: a client needs a transport and a peer other than 0");

    // Data files: an error stops the program in every build (R-DATA-1: a game must not run on half-read data)
    dataReadOnly = setup.sharedDataReadOnly;
    if (setup.sharedData)
        dataStore = setup.sharedData;
    else
    {
        DataTypeList types;
        if (setup.dataTypes) setup.dataTypes(types);
        dataStore = std::make_shared<DataStore>(std::move(types), setup.dataDirectory);
        std::string error;
        if (!dataStore->load(error)) SAIGA_EXIT_ERROR("data files: " + error);
    }

    prof.enabled = setup.profiling;
    setupProfiler();

    // Before the state entities: the host's replication consumer sees their creation (S16)
    repl = std::make_unique<Replication>(*this);
    createStateEntities();
    // The game's meshes and materials by name (S19b); a sink set later receives them
    if (setup.renderAssets) setup.renderAssets(*this);
}

void World::setupProfiler()
{
    stepScope = prof.scope("step");
    for (int p = kFirstPhase; p <= kLastPhase; ++p)
    {
        const std::string phasePath = std::string("step/phase:") + phaseName(Phase(p));
        phaseScopes[size_t(p)]      = prof.scope(phasePath);
        for (const SystemList::System& s : systems.systems(Phase(p)))
            systemScopes[size_t(p)].push_back(prof.scope(phasePath + "/system:" + s.name));
        flushScopes[size_t(p)] = prof.scope(phasePath + "/commands");
    }
    // The extract after phase 8's systems, replication and interpolation (S19b)
    extractScope    = prof.scope(std::string("step/phase:") + phaseName(Phase::Extract) + "/extract");
    entitiesCounter = prof.counter("entities");
    objectsCounter  = prof.counter("game_objects");
    updatedCounter  = prof.counter("objects_updated");  // behaviourSystem adds to it
    commandsCounter = prof.counter("commands_applied");
    recordsCounter  = prof.counter("structural_records");
    gridCounter     = prof.counter("grid_items");
    if (net)
    {
        counterSentBytes     = prof.counter("net/sent_bytes");
        counterReceivedBytes = prof.counter("net/received_bytes");
        counterMessagesIn    = prof.counter("net/messages_in");
        counterMessagesOut   = prof.counter("net/messages_out");
        counterRejected      = prof.counter("net/rejected");
    }
}

void World::createStateEntities()
{
    // The world-state entity and one space-state entity per space (SpaceId order, S21) exist when components are
    // registered for them (§7.8)
    bool anyGlobal = false, anySpace = false;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        anyGlobal = anyGlobal || reg.isGlobal(ComponentId(id));
        anySpace  = anySpace || reg.isSpaceState(ComponentId(id));
    }
    if (anyGlobal) worldState = makeStateEntity(Registry::kWorldState, 0, nullptr);
    if (anySpace)
        for (size_t space = 0; space < spaceCount(); ++space)
            spaceStates.push_back(makeStateEntity(Registry::kSpaceState, SpaceId(space), nullptr));
}

Entity World::makeStateEntity(uint8_t flag, SpaceId space, StateDefaults* defaults)
{
    // Not World::create: the loader creates state entities outside a step's rules
    Entity e;
    {
        auto record = log.prepare(1);
        e           = reg.create();
        record.append(e, Transform::id, true);
    }
    reg.setFlags(e, uint8_t(reg.flags(e) | flag));
    // A space-state entity's identity is its Transform::space (writer 6); the world-state entity stays in space 0
    if (flag & Registry::kSpaceState) reg.get<Transform>(e).setSpaceInternal(space);
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        const bool state =
            (flag & Registry::kWorldState) ? reg.isGlobal(ComponentId(id)) : reg.isSpaceState(ComponentId(id));
        if (state) addStateComponent(e, ComponentId(id), defaults);
    }
    return e;
}

void World::addStateComponent(Entity e, ComponentId id, StateDefaults* defaults)
{
    auto record = log.prepare(1);
    if (defaults && (*defaults)[id])
        reg.addStaged(e, id, *(*defaults)[id]);
    else
        reg.addDefault(e, id);
    record.append(e, id, true);
}

// ---- game objects and gameplay
// -------------------------------------------------------------------------------------------

void World::addDefaultNow(Entity e, ComponentId id)
{
    SAIGA_ASSERT(!inStep(), "addDefaultNow: between steps");
    SAIGA_ASSERT(id != Transform::id && id != Behaviour::id && id != Persistent::id && id != Parent::id,
                 "addDefaultNow: this component has its own path");
    addDefaultInternal(e, id);
}

void World::removeByIdNow(Entity e, ComponentId id)
{
    SAIGA_ASSERT(!inStep(), "removeByIdNow: between steps");
    if (id == Parent::id) return removeNow<Parent>(e);
    if (id == Persistent::id) return removeNow<Persistent>(e);
    if (id == Behaviour::id) return removeNow<Behaviour>(e);  // the object may be running: retired
    if (id == NetOwner::id) return removeNow<NetOwner>(e);    // the owner rules of add/remove
    removeInternal(e, id);
}

bool World::reparentNow(Entity child, Entity parent)
{
    SAIGA_ASSERT(!inStep(), "reparentNow: between steps");
    checkNotParallel("reparentNow");
    if (!reg.alive(child) || !reg.has<Parent>(child))
    {
        // Refused in every build
        SAIGA_ASSERT(false, "reparentNow: the entity is not an alive child");
        return false;
    }
    if (!acceptParent(child, parent)) return false;
    // The owner before (the old root) against the owner after (the new root): no intermediate owner change
    if (netChangeRefused(child, Parent::id, true, parent)) return false;
    auto record                       = log.prepare(2);
    const std::vector<Entity> subtree = subtreeToFollow(child, parent);
    reserveChildSlot(parent);
    Parent& p        = reg.get<Parent>(child);
    const Entity old = p.entity;
    p.entity         = parent;
    changeChildCount(old, -1);
    changeChildCount(parent, +1);
    hierarchyDirty  = true;
    ownerCacheDirty = true;
    // As a remove and an add (the replication sends one Hierarchy message)
    record.append(child, Parent::id, false);
    record.append(child, Parent::id, true);
    followParentSpace(child, subtree);
    return true;
}

Entity World::spawnByClass(const ClassList::ClassInfo& info, SpaceId space, const Vec3& position,
                           const std::function<void(Entity)>& initial)
{
    SAIGA_ASSERT(!inStep(), "spawnByClass: between steps");
    const Entity e = beginSpawn(space);
    if (!e.valid()) return e;
    try
    {
        finishSpawn(e, info, space, position);
        if (initial) initial(e);
    }
    catch (...)
    {
        abandonSpawn(e);
        throw;
    }
    startObject(e);
    return e;
}

void World::runStepListeners()
{
    // Listeners may add listeners (kept) and remove themselves (return false). A throwing listener is removed; the
    // others are still notified of this (committed) step, then the first exception is rethrown.
    std::vector<std::pair<int, std::function<bool()>>> current;
    current.swap(stepListeners);
    removedStepListeners.clear();
    dispatchingStepListeners = true;
    // Their sends between steps go out together when the dispatch ends (also when a listener threw)
    beginSendBatch();
    std::exception_ptr first;
    for (auto& l : current)
    {
        if (std::find(removedStepListeners.begin(), removedStepListeners.end(), l.first) != removedStepListeners.end())
            continue;
        try
        {
            if (l.second()) stepListeners.push_back(std::move(l));
        }
        catch (...)
        {
            if (!first) first = std::current_exception();
        }
    }
    removedStepListeners.clear();
    dispatchingStepListeners = false;
    endSendBatch();
    if (first) std::rethrow_exception(first);
}

Entity World::beginSpawn(SpaceId space)
{
    checkNotParallel("spawn");
    // An unknown space is refused in every build too (no entity, no hooks; PR review, S21)
    if (space >= spaceCount())
    {
        SAIGA_ASSERT(false, "spawnIn: unknown space " + std::to_string(space));
        return Entity{};
    }
    // A replicated game object only arrives on a client (§13.5); refused in every build (no entity, no hooks)
    if (!isHost())
    {
        SAIGA_ASSERT(false, "spawn<T> on a client (game objects arrive through replication)");
        return Entity{};
    }
    return create();
}

void World::finishSpawn(Entity e, const ClassList::ClassInfo& info, SpaceId space, const Vec3& position)
{
    Transform& t = reg.get<Transform>(e);
    t.setSpaceInternal(space);
    t.setPosition(position);
    Behaviour b;
    b.object        = info.factory();
    b.object->owner = this;
    b.object->self  = e;
    b.classId       = info.id;
    addNow<Behaviour>(e, std::move(b));
}

void World::startObject(Entity e)
{
    // configure or onSpawn may destroy the object (a command during the flush applies at once)
    if (!reg.alive(e)) return;
    Behaviour& b = reg.get<Behaviour>(e);
    assignTickInterval(*this, e, b, true);
    GameObject* o = b.object.get();
    // The object stays valid while its hooks run, also if they destroy its entity (released afterwards)
    hookDepth++;
    try
    {
        o->configure();
        if (reg.alive(e) && reg.has<Behaviour>(e)) o->onSpawn();
    }
    catch (...)
    {
        hookDepth--;
        // A half-started object must not tick: the entity is destroyed (without its onDestroy)
        abandonSpawn(e);
        throw;
    }
    hookDepth--;
    releaseRetired();
}

void World::abandonSpawn(Entity e)
{
    if (!reg.alive(e)) return;
    // The whole subtree without hooks (a hook of a child created by the failed configure must not stop the cleanup)
    for (Entity x : collectDoomed(e))
        if (Behaviour* b = reg.tryGet<Behaviour>(x)) b->destroying = true;
    // At once, also inside a destroy (the queue would be dropped by the exception that is propagating)
    destroyOne(e);
}

void World::queueAction(const PlayerAction& action)
{
    checkNotParallel("queueAction");
    if (!reg.alive(action.player) || !reg.has<Player>(action.player))
    {
        // Refused in every build
        SAIGA_ASSERT(false, "queueAction: the player is not alive or has no Player component");
        return;
    }
    if (!isHost())
    {
        // A client sends the actions of the players it owns (§9.3); refused in every build otherwise
        if (ownerOf(action.player) != selfPeer)
        {
            SAIGA_ASSERT(false, "queueAction: a client queues actions only for the players it owns");
            return;
        }
        repl->sendPlayerAction(action);
        return;
    }
    actions.push_back(QueuedAction{reg.read<Player>(action.player).peer, actionSequence++, action});
}

void World::setOwner(Entity root, uint8_t peer)
{
    checkNotParallel("setOwner");
    // Refused in every build
    if (!isHost())
    {
        SAIGA_ASSERT(false, "setOwner on a client (ownership is set by the host)");
        return;
    }
    if (!reg.alive(root) && !reg.reserved(root))
    {
        SAIGA_ASSERT(false, "setOwner: the entity is not alive");
        return;
    }
    if (reg.alive(root))
    {
        if (reg.has<Parent>(root))
        {
            SAIGA_ASSERT(false, "setOwner: the entity is a child (ownership is defined on roots, §13.3)");
            return;
        }
        if (repl->spawned(root))
        {
            SAIGA_ASSERT(false, "setOwner: the entity's spawn was sent already (no ownership transfer, §13.3)");
            return;
        }
        if (childCount(root) > 0)
            for (Entity x : collectDoomed(root))
                if (repl->spawned(x))
                {
                    // A spawned descendant would change its owner through its root (any NetOwner change, as add/remove)
                    SAIGA_ASSERT(false,
                                 "setOwner: a descendant's spawn was sent already (no ownership transfer, §13.3)");
                    return;
                }
        if (const Player* p = reg.tryRead<Player>(root); p && p->peer != peer)
        {
            SAIGA_ASSERT(false, "setOwner: Player.peer " + std::to_string(p->peer) + " differs from the owner " +
                                    std::to_string(peer));
            return;
        }
        if (NetOwner* o = reg.tryGet<NetOwner>(root))
        {
            // Applied at once, also inside a system: the cache is rebuilt now (the system's parallel loops read it)
            o->peer = peer;
            rebuildOwnerCache();
            return;
        }
    }
    add<NetOwner>(root, NetOwner{peer});
}

void World::ownerTeleport(Entity root, const Vec3& position, const Quat& rotation)
{
    checkNotParallel("ownerTeleport");
    if (!isHost())
    {
        SAIGA_ASSERT(false, "ownerTeleport on a client (a host command)");
        return;
    }
    if (!reg.alive(root) || reg.has<Parent>(root))
    {
        SAIGA_ASSERT(false, "ownerTeleport: the entity is not an alive root");
        return;
    }
    moveToSpace(root, reg.read<Transform>(root).space(), position, rotation);
}

void World::moveToSpace(Entity e, SpaceId space, const Vec3& position, const Quat& rotation)
{
    checkNotParallel("moveToSpace");
    // Code errors (the world API checks them and returns errors instead); refused without asserts too
    if (!reg.alive(e))
    {
        SAIGA_ASSERT(false, "moveToSpace: the entity is not alive (or reserved)");
        return;
    }
    if (reg.has<Parent>(e))
    {
        SAIGA_ASSERT(false, "moveToSpace: the entity is a child (its parent decides)");
        return;
    }
    if (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState))
    {
        SAIGA_ASSERT(false, "moveToSpace: a world-state or space-state entity");
        return;
    }
    if (space >= spaceCount())
    {
        SAIGA_ASSERT(false, "moveToSpace: unknown space " + std::to_string(space));
        return;
    }
    if (clientSubtreeReplicated(e))
    {
        // R-NET-2: clients never move replicated entities (the owner applies the host's OwnerCommand), also not as the
        // descendants of a local root. Regression (PR review, S21): only the root was checked
        SAIGA_ASSERT(false, "moveToSpace: a client cannot move a replicated entity (or a subtree with one)");
        return;
    }
    if (isHost())
        repl->ownerMove(e, space, position, rotation);
    else
        teleportTo(e, space, position, rotation);
}

void World::requestTransition(Entity e, Entity link)
{
    checkNotParallel("requestTransition");
    if (!isHost())
    {
        SAIGA_ASSERT(false, "requestTransition on a client (a host command)");
        return;
    }
    if (!reg.alive(e))
    {
        SAIGA_ASSERT(false, "requestTransition: the entity is not alive (or reserved)");
        return;
    }
    if (reg.has<Parent>(e))
    {
        SAIGA_ASSERT(false, "requestTransition: the entity is a child (its parent decides)");
        return;
    }
    if (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState))
    {
        SAIGA_ASSERT(false, "requestTransition: a world-state or space-state entity");
        return;
    }
    if (!reg.alive(link) || !reg.has<SpaceLink>(link))
    {
        SAIGA_ASSERT(false, "requestTransition: the link is not alive or has no SpaceLink");
        return;
    }
    const SpaceLink& l = reg.read<SpaceLink>(link);
    // Data errors (a hand-written or tool-edited link, a second request): refused in every build
    if (l.target >= spaceCount())
    {
        SAIGA_LOG_E("world", e, "requestTransition refused: the link e%u:%u targets the unknown space %d", link.index,
                    link.generation, int(l.target));
        return;
    }
    if (reg.has<PendingTransition>(e))
    {
        SAIGA_LOG_E("world", e, "requestTransition refused: a transition is pending already (the first request wins)");
        return;
    }
    PendingTransition p;
    p.link     = link;
    p.target   = l.target;
    p.position = l.arrival;
    p.rotation = l.arrivalRotation;
    if (!deferring())
    {
        if (inStep()) transitionsRequestedInStep.push_back(e);
        addNow<PendingTransition>(e, std::move(p));
        return;
    }
    record(makeCommand(
        [e, p](World& w) mutable
        {
            // Re-checked at the flush: the entity may have died or got a parent (or a request) in the same phase
            if (!w.reg.alive(e) || w.reg.has<Parent>(e) || w.reg.has<PendingTransition>(e))
            {
                SAIGA_LOG_E("world", e,
                            "requestTransition dropped at the flush: not an alive root without a pending "
                            "transition any more");
                return;
            }
            w.transitionsRequestedInStep.push_back(e);
            w.addNow<PendingTransition>(e, std::move(p));
        }));
}

void World::setSubtreeSpace(Entity root, SpaceId space)
{
    for (Entity x : collectSubtree(root))
    {
        Transform& t = reg.get<Transform>(x);
        if (t.space() == space) continue;
        t.setSpaceInternal(space);
        repl->noteWritten(x, Transform::id);
    }
}

void World::teleportTo(Entity root, SpaceId space, const Vec3& position, const Quat& rotation)
{
    Transform& t = reg.get<Transform>(root);
    if (space != t.space()) setSubtreeSpace(root, space);
    reg.get<Transform>(root).teleport(position, rotation);
    repl->noteWritten(root, Transform::id);
}

std::vector<Entity> World::subtreeToFollow(Entity child, Entity parent) const
{
    if (reg.read<Transform>(child).space() == reg.read<Transform>(parent).space()) return {};
    return collectSubtree(child);
}

void World::followParentSpace(Entity child, const std::vector<Entity>& subtree)
{
    if (subtree.empty()) return;
    const SpaceId space    = reg.read<Transform>(reg.read<Parent>(child).entity).space();
    const bool renderPoses = reg.registered(RenderPose::id);
    for (Entity x : subtree)
    {
        Transform& t = reg.get<Transform>(x);
        if (t.space() == space) continue;
        t.setSpaceInternal(space);
        repl->noteWritten(x, Transform::id);
        // Regression (PR review, S21): a RenderPose kept the old space (the extract drew the entity there) when no
        // ancestor's RenderPose rewrote it, and its buffer's old-space samples would rewrite it again
        if (renderPoses)
            if (RenderPose* rp = reg.tryGet<RenderPose>(x))
            {
                rp->position = t.position();
                rp->rotation = t.rotation();
                rp->space    = space;
                rp->flags    = uint8_t(rp->flags | RenderPose::kChanged | RenderPose::kTeleported);
                repl->buffers.erase(x.index);
            }
    }
    // At its current pose: phase 6 writes the parent-relative one; the flag keeps the extract and the interpolation
    // from blending across spaces
    Transform& c = reg.get<Transform>(child);
    c.teleport(c.position(), c.rotation());
    repl->noteWritten(child, Transform::id);
}

void World::rebuildOwnerCacheChecked() const
{
    SAIGA_ASSERT(!reg.inParallelEach(), "ownerOf/ownedBy: the owner cache is stale inside parallelEach (R-PAR-1)");
    rebuildOwnerCache();
}

const std::vector<uint8_t>& World::OwnerTableOfWorld::table() const
{
    if (world->ownerCacheDirty) world->rebuildOwnerCacheChecked();
    return world->ownerCache;
}

bool World::clientSubtreeReplicated(Entity e) const
{
    if (isHost() || !reg.alive(e)) return false;
    for (Entity x : collectDoomed(e))
        if (repl->netIdOf(x) != 0) return true;
    return false;
}

void World::rebuildOwnerCache() const
{
    ownerCache.assign(size_t(reg.slotCount()), uint8_t(0));
    ownerCacheDirty = false;
    if (!reg.registered(NetOwner::id)) return;
    Registry& r = const_cast<Registry&>(reg);
    bool any    = false;
    r.view<const NetOwner>().each(
        [&](Entity e, const NetOwner& o)
        {
            ownerCache[e.index] = o.peer;
            any                 = any || o.peer != 0;
        });
    if (!any || !reg.registered(Parent::id)) return;
    // Descendants inherit the root's owner (bounded walk: a broken registry cannot loop)
    const uint32_t max = reg.slotCount();
    r.view<const Parent>().each(
        [&](Entity e, const Parent& p)
        {
            Entity a = p.entity;
            for (uint32_t depth = 0; depth <= max && reg.alive(a); ++depth)
            {
                const Parent* up = reg.tryRead<Parent>(a);
                if (!up) break;
                a = up->entity;
            }
            if (reg.alive(a)) ownerCache[e.index] = ownerCache[a.index];
        });
}

void World::addDefaultInternal(Entity e, ComponentId id)
{
    auto record = log.prepare(1);
    reg.addDefault(e, id);
    record.append(e, id, true);
    if (id == NetOwner::id) ownerCacheDirty = true;
}

void World::removeInternal(Entity e, ComponentId id)
{
    if (id == Parent::id) return removeNow<Parent>(e);
    if (id == Persistent::id) return removeNow<Persistent>(e);
    if (id == Behaviour::id) return removeNow<Behaviour>(e);
    auto record = log.prepare(1);
    reg.removeById(e, id);
    record.append(e, id, false);
    if (id == NetOwner::id) ownerCacheDirty = true;
}

void World::deliverActions()
{
    // (peer, sequence) order, exactly once: the queue becomes pending ActionEvents, readable in this step
    std::stable_sort(actions.begin(), actions.end(), [](const QueuedAction& a, const QueuedAction& b)
                     { return a.peer != b.peer ? a.peer < b.peer : a.sequence < b.sequence; });
    for (const QueuedAction& a : actions) events.emit(ActionEvent{a.action.player, a.action.action, a.action.aim});
    actions.clear();
}

PersistentId World::makePersistent(Entity e)
{
    checkNotParallel("makePersistent");
    // The client's counter would collide with the host's IDs; refused in every build
    if (!isHost())
    {
        SAIGA_ASSERT(false, "makePersistent on a client (Persistent IDs come from the host)");
        return PersistentId{};
    }
    if (loadHookPhase && !checkLoadHook("makePersistent", e, Persistent::id)) return PersistentId{};
    if (!reg.alive(e) && !reg.reserved(e))
    {
        // Refused in every build
        SAIGA_ASSERT(false, "makePersistent: the entity is not alive");
        return PersistentId{};
    }
    if (reg.alive(e))
        if (const Persistent* p = reg.tryRead<Persistent>(e)) return p->persistentId;
    // Assigned earlier in this phase (the component is added at the flush): the same ID (idempotent)
    for (const auto& pending : pendingPersistent)
        if (pending.first == e) return pending.second;
    const PersistentId id{++persistentCounter};
    Persistent value;
    value.persistentId = id;
    if (deferring() || (flushing && reg.reserved(e)))
    {
        pendingPersistent.push_back({e, id});
        record(makeCommand([e, value](World& w) mutable { w.addNow<Persistent>(e, std::move(value)); }));
    }
    else
        addNow<Persistent>(e, std::move(value));
    return id;
}

Entity World::resolve(PersistentId id) const
{
    auto it = persistentIndex.find(id.value);
    if (it == persistentIndex.end() || !reg.alive(it->second)) return Entity{};
    return it->second;
}

// ---- spaces and terrain (S20)

const SpaceTemplate& World::spaceTemplate(SpaceId space) const
{
    return terrainSys->spaceTemplate(space);
}

size_t World::spaceCount() const
{
    return terrainSys->spaceCount();
}

uint64_t World::spaceSeed(SpaceId space) const
{
    SAIGA_ASSERT(space < terrainSys->spaceCount(), "World::spaceSeed: unknown space");
    return terrainSys->spaceSeed(space);
}

double World::terrainHeight(SpaceId space, double x, double z) const
{
    return terrainSys->height(space, x, z);
}

double World::spaceReadiness(SpaceId space, const Vec3& camera, double radius) const
{
    SAIGA_ASSERT(space < spaceCount(), "World::spaceReadiness: unknown space");
    // S21: the terrain's (S20, D6); S23b adds the content sets, S27 far content
    return terrainSys->readiness(space, camera, radius);
}

World::~World()
{
    // Observers (the world API, waiting tools) stop using the world
    for (auto& l : destroyListeners) l.second();
    // Before the structural log (a member declared after it): its consumer goes; the sink is not called
    extractor.reset();
}

// ---- spatial queries
// ----------------------------------------------------------------------------------------------

std::vector<Entity> World::queryRadius(SpaceId space, const Vec3& center, double radius,
                                       const ComponentMask& mask) const
{
    std::vector<Entity> out;
    queryRadius(space, center, radius, mask, out);
    return out;
}

void World::queryRadius(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask,
                        std::vector<Entity>& out) const
{
    out.clear();
    forEachWithin(space, center, radius, mask, [&](Entity e, const Vec3&) { out.push_back(e); });
    std::sort(out.begin(), out.end());
}

// ---- structural changes
// ----------------------------------------------------------------------------------------------

Entity World::create()
{
    checkNotParallel("create");
    // A refused create still returns an entity handle the caller expects: refused only with asserts
    if (loadHookPhase) checkLoadHook("create", Entity{}, Transform::id);
    if (deferring())
    {
        // Reserved now (views skip it), alive when the command is applied; reserving never moves a Transform
        // Everything the abort cleanup needs is allocated now (it must not allocate while an exception propagates)
        // (the registry keeps room for releasing every reserved entity)
        // Geometric growth: reserving exactly size + 1 reallocated on every create (quadratic in creates per phase)
        if (pendingReserved.capacity() == pendingReserved.size())
            pendingReserved.reserve(std::max<size_t>(16, 2 * pendingReserved.capacity()));
        Entity e = reg.reserve();
        pendingReserved.push_back(e);
        record(makeCommand([e](World& w) { w.activateNow(e); }));
        return e;
    }
    auto record = log.prepare(1);  // the record cannot be lost after the change (append does not allocate)
    Entity e    = reg.create();
    record.append(e, Transform::id, true);
    return e;
}

Entity World::createIn(SpaceId space)
{
    SAIGA_ASSERT(!inStep(), "createIn: between steps");
    SAIGA_ASSERT(space < spaceCount(), "createIn: unknown space " + std::to_string(space));
    const Entity e = create();
    reg.get<Transform>(e).setSpaceInternal(space);
    return e;
}

void World::destroy(Entity e)
{
    checkNotParallel("destroy");
    if (loadHookPhase && !checkLoadHook("destroy", e, Transform::id)) return;
    if (!isHost() && repl->netIdOf(e) != 0)
    {
        // Refused in every build: replicated entities are destroyed by the host (§13.5)
        SAIGA_ASSERT(false, "destroy: a client cannot destroy a replicated entity");
        return;
    }
    if (clientSubtreeReplicated(e))
    {
        // A replicated entity parented under a local one would be destroyed by the cascade
        SAIGA_ASSERT(false, "destroy: a client cannot destroy a replicated entity (a descendant)");
        return;
    }
    if (reg.alive(e) && (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState)))
    {
        // Refused in every build (the world state must stay; asserts in builds with asserts)
        SAIGA_ASSERT(false, "destroy: world-state and space-state entities cannot be destroyed");
        return;
    }
    // During a flush an entity reserved later in the phase is not active yet: the destroy follows its activation
    if (deferring() || (flushing && reg.reserved(e)))
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), "destroy: the entity is not alive");
        // Dropped if the entity is gone when the command is applied (e.g. destroyed with its parent)
        record(makeCommand(
            [e](World& w)
            {
                if (!w.reg.alive(e)) return;
                // The subtree may have changed since (a replica linked under it earlier in the flush)
                if (w.clientSubtreeReplicated(e))
                {
                    SAIGA_ASSERT(false, "destroy: a client cannot destroy a replicated entity (a descendant)");
                    return;
                }
                w.destroyNow(e);
            }));
        return;
    }
    destroyNow(e);
}

void World::activateNow(Entity e)
{
    auto record = log.prepare(1);
    reg.activate(e);
    record.append(e, Transform::id, true);
}

std::vector<Entity> World::collectDoomed(Entity e) const
{
    return collectSubtree(e);
}

std::vector<Entity> World::collectSubtree(Entity e) const
{
    // The entity and its descendants (§7.4), parents before children. One pass over the Parent pool builds the
    // (parent index, child) list, sorted for a deterministic order; the visited set stops at cycles (add<Parent>
    // rejects them, this keeps destroy finite regardless).
    std::vector<Entity> doomed{e};
    if (childCount(e) == 0) return doomed;
    std::vector<std::pair<uint32_t, Entity>> links;
    const_cast<Registry&>(reg).view<const Parent>().each([&](Entity child, const Parent& p)
                                                         { links.push_back({p.entity.index, child}); });
    std::sort(links.begin(), links.end(), [](const std::pair<uint32_t, Entity>& a, const std::pair<uint32_t, Entity>& b)
              { return a.first != b.first ? a.first < b.first : a.second.index < b.second.index; });
    std::vector<bool> visited(reg.slotCount(), false);
    visited[e.index] = true;
    for (size_t i = 0; i < doomed.size(); ++i)
    {
        const Entity parent = doomed[i];
        auto it =
            std::lower_bound(links.begin(), links.end(), parent.index,
                             [](const std::pair<uint32_t, Entity>& l, uint32_t index) { return l.first < index; });
        for (; it != links.end() && it->first == parent.index; ++it)
        {
            const Entity child = it->second;
            if (childCount(parent) == 0) break;
            if (visited[child.index] || reg.read<Parent>(child).entity != parent) continue;
            visited[child.index] = true;
            doomed.push_back(child);
        }
    }
    return doomed;
}

void World::destroyNow(Entity e)
{
    // A destroy requested while a destroy runs (an onDestroy hook) is queued and applied after it, iteratively: no
    // recursion depth per chained destroy
    if (destroyDepth > 0)
    {
        destroyQueue.push_back(e);
        return;
    }
    destroyQueue.push_back(e);
    try
    {
        for (size_t i = 0; i < destroyQueue.size(); ++i)
            if (reg.alive(destroyQueue[i])) destroyOne(destroyQueue[i]);
    }
    catch (...)
    {
        destroyQueue.clear();
        throw;
    }
    destroyQueue.clear();
}

void World::destroyOne(Entity e)
{
    std::vector<Entity> doomed = collectDoomed(e);
    // The game objects first, children before parents, while all components are present. A hook may destroy other
    // entities (queued, applied after this destroy), spawn new ones or attach entities to a doomed parent: the
    // descendants are collected again after the hooks, and the new ones are destroyed without their onDestroy (a
    // hook that keeps adding children cannot loop).
    std::vector<Entity> hooked;
    destroyDepth++;
    hookDepth++;
    try
    {
        for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
        {
            if (!reg.alive(*it)) continue;
            Behaviour* b = reg.tryGet<Behaviour>(*it);
            if (!b || !b->object || b->destroying) continue;
            b->destroying = true;
            hooked.push_back(*it);
            b->object->onDestroy();
        }
        if (reg.alive(e)) doomed = collectDoomed(e);
        for (Entity x : doomed)
            if (Behaviour* b = reg.tryGet<Behaviour>(x)) b->destroying = true;
    }
    catch (...)
    {
        // Nothing was destroyed: the entities can be destroyed again later (their onDestroy runs again then)
        for (Entity x : hooked)
            if (Behaviour* b = reg.alive(x) ? reg.tryGet<Behaviour>(x) : nullptr) b->destroying = false;
        destroyDepth--;
        hookDepth--;
        throw;
    }
    destroyDepth--;
    hookDepth--;
    for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
    {
        const Entity x = *it;
        // A hook (onDestroy) may have destroyed it already
        if (!reg.alive(x)) continue;
        // Snapshot, reserve the log records and the room for the retired object, then destroy (the only step that
        // can throw, before any change); the bookkeeping is committed afterwards without allocation
        const ComponentMask m = reg.entityMask(x);
        size_t records        = 0;
        for (size_t id = 0; id < kMaxComponents; ++id) records += m.test(ComponentId(id)) ? 1 : 0;
        auto prepared = log.prepare(records);
        retired.reserve(retired.size() + 1);
        const Parent* p        = reg.tryRead<Parent>(x);
        const Entity oldParent = p ? p->entity : Entity{};
        const Persistent* pid  = reg.tryRead<Persistent>(x);
        const uint64_t oldId   = pid ? pid->persistentId.value : 0;
        // The game object may still be running (a hook that destroyed its own entity): it is released when no hook
        // runs any more
        if (Behaviour* b = reg.tryGet<Behaviour>(x); b && b->object) retired.push_back(std::move(b->object));
        reg.destroy(x);
        if (oldId) persistentIndex.erase(oldId);
        if (oldParent.valid()) changeChildCount(oldParent, -1);
        if (x.index < children.size()) children[x.index] = 0;
        // One removed record per component in ID order, Transform first
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) prepared.append(x, ComponentId(id), false);
        if (m.test(Parent::id) || doomed.size() > 1) hierarchyDirty = true;
        if (m.test(Parent::id) || m.test(NetOwner::id) || doomed.size() > 1) ownerCacheDirty = true;
    }
    // An entity whose onDestroy ran but that a hook released from the cascade (e.g. removed its Parent) lives on:
    // its flag is reset, so its real destroy later calls onDestroy again
    for (Entity x : hooked)
        if (Behaviour* b = reg.alive(x) ? reg.tryGet<Behaviour>(x) : nullptr) b->destroying = false;
    releaseRetired();
}

void World::releaseRetired()
{
    // Game objects of destroyed entities (or removed Behaviours) live until no hook runs any more
    if (hookDepth == 0) retired.clear();
}

bool World::rebuildHierarchyState()
{
    // Built separately and validated before it replaces the old counts: every Parent must name an alive entity (a
    // stale or out-of-range handle from a direct registry fill is rejected in every build)
    std::vector<uint32_t> counts(size_t(reg.slotCount()), 0);
    bool ok = true;
    if (reg.registered(Parent::id))
        reg.view<const Parent>().each(
            [&](Entity, const Parent& p)
            {
                if (!reg.alive(p.entity))
                    ok = false;
                else
                    counts[p.entity.index]++;
            });
    if (!ok) return false;
    // No cycles: every chain must reach a root within slotCount links
    if (reg.registered(Parent::id))
    {
        const uint32_t max = reg.slotCount();
        reg.view<const Parent>().each(
            [&](Entity, const Parent& p)
            {
                Entity a = p.entity;
                for (uint32_t depth = 0; ok; ++depth)
                {
                    const Parent* up = reg.tryRead<Parent>(a);
                    if (!up) break;
                    if (depth > max || !reg.alive(up->entity)) ok = false;
                    a = up->entity;
                }
            });
    }
    if (!ok) return false;
    children.swap(counts);
    hierarchyDirty  = true;
    ownerCacheDirty = true;
    return true;
}

uint32_t World::childCount(Entity e) const
{
    return e.index < children.size() ? children[e.index] : 0;
}

void World::reserveChildSlot(Entity parent)
{
    if (parent.index >= children.size()) children.resize(size_t(reg.slotCount()), 0);
}

void World::changeChildCount(Entity parent, int delta)
{
    reserveChildSlot(parent);
    SAIGA_ASSERT(delta > 0 || children[parent.index] > 0, "child count underflow");
    children[parent.index] = uint32_t(int64_t(children[parent.index]) + delta);
}

bool World::acceptParent(Entity child, Entity parent)
{
    // World-state entities are never children (a parent's destroy would take them along); refused in every build
    if (reg.flags(child) & (Registry::kWorldState | Registry::kSpaceState))
    {
        SAIGA_ASSERT(false, "add<Parent>: world-state and space-state entities cannot have a parent");
        return false;
    }
    if (!reg.alive(parent))
    {
        // During a flush the parent may have been destroyed earlier in the same flush: the child follows it
        // Between steps a dead parent is an error (refused without asserts); during a flush the parent may have been
        // destroyed earlier in the same flush: the child follows it
        SAIGA_ASSERT(flushing, "add<Parent>: the parent is not alive");
        // A client does not destroy replicas (§13.5): the link is dropped and the child stays
        if (flushing && !clientSubtreeReplicated(child)) destroyNow(child);
        return false;
    }
    // No cycles: the child must not be an ancestor of the parent (or the parent itself). An acyclic chain has fewer
    // links than there are slots, so a longer walk means an existing cycle: refused as well (never accepted early).
    Entity a           = parent;
    const uint32_t max = reg.slotCount();
    for (uint32_t depth = 0;; ++depth)
    {
        SAIGA_ASSERT(a != child, "add<Parent>: the parent is the child or one of its descendants (a cycle)");
        if (a == child || depth > max) return false;
        const Parent* p = reg.tryRead<Parent>(a);
        if (!p) return true;
        a = p->entity;
    }
}

void World::abortStep()
{
    // The outbox entries of the interrupted phase are dropped like its commands (in place: no allocation); those of the
    // finished phases stay and are sent with the next flush
    const Phase aborted = phase;
    outbox.erase(
        std::remove_if(outbox.begin(), outbox.end(), [aborted](const OutboxEntry& e) { return e.phase == aborted; }),
        outbox.end());
    // The kept entries belong to no phase any more (a later abort in their phase must not drop them)
    for (OutboxEntry& e : outbox) e.phase = Phase::None;
    // The world leaves the step first (whatever happens below, the next step can run)
    phase              = Phase::None;
    flushing           = false;
    const bool restore = stepEventsBegun;
    stepEventsBegun    = false;
    // The commands of the interrupted phase are dropped, the entities they would have activated released, and the
    // events of the step are restored: the readable ones become pending again, the ones the aborted step emitted are
    // dropped like its commands. None of this allocates (the capacity for the releases was reserved when the entities
    // were reserved; the event restore is a swap), so the cleanup cannot fail while the exception propagates.
    commands.clear();
    if (restore) events.abortStep();
    // The replication bookkeeping staged in the aborted phase is dropped (S16)
    repl->abort(aborted);
    // The extract's staged pairs too (S19b)
    extractor->abort();
    pendingPersistent.clear();
    for (Entity e : pendingReserved)
        if (reg.reserved(e)) reg.release(e);
    pendingReserved.clear();
}

void World::record(std::unique_ptr<Command> c)
{
    commands.push_back(std::move(c));
}

void World::flushCommands()
{
    // Changes made by a command apply immediately (flushing is set, so nothing is recorded during a flush)
    flushing = true;
    for (size_t i = 0; i < commands.size(); ++i)
    {
        std::unique_ptr<Command> c = std::move(commands[i]);
        c->apply(*this);
    }
    commands.clear();
    pendingReserved.clear();
    pendingPersistent.clear();
    flushing = false;
    releaseRetired();
}

bool World::netChangeRefused(Entity e, ComponentId c, bool add, Entity parent)
{
    if (!isHost() || !reg.alive(e)) return false;
    if (c == ComponentId(EngineComponent::NetOwner) && !repl->spawned(e) && childCount(e) > 0)
    {
        // A NetOwner change of an unspawned root changes the owner of its spawned descendants too
        for (Entity x : collectDoomed(e))
            if (repl->spawned(x))
            {
                SAIGA_ASSERT(false, std::string(add ? "add<" : "remove<") +
                                        "NetOwner>: it would change the owner of a spawned descendant (no transfer "
                                        "after the spawn, §13.3)");
                return true;
            }
        return false;
    }
    if (c == ComponentId(EngineComponent::Parent))
    {
        // The moved subtree's owner before and after (a child has no NetOwner: without a parent it belongs to the
        // host); refused when it changes for a spawned entity in the subtree, also below an unspawned root
        const uint8_t before = ownerOf(e);
        const uint8_t after  = add ? (reg.alive(parent) ? ownerOf(parent) : before) : uint8_t(0);
        if (after == before) return false;
        bool spawned = repl->spawned(e);
        if (!spawned && childCount(e) > 0)
            for (Entity x : collectDoomed(e))
                if (repl->spawned(x))
                {
                    spawned = true;
                    break;
                }
        if (!spawned) return false;
    }
    else if (!repl->spawned(e))
        return false;  // NetOwner of an unspawned root without spawned descendants
    // Any NetOwner change of a spawned root is a transfer
    SAIGA_ASSERT(false, std::string(add ? "add<" : "remove<") + reg.componentInfo(c).name +
                            ">: it would change the owner of a spawned entity (no transfer after the spawn, §13.3)");
    return true;
}

void World::checkNotParallel(const char* operation) const
{
    SAIGA_ASSERT(!reg.inParallelEach(), std::string(operation) + " inside parallelEach (R-PAR-1)");
}

// ---- steps
// -----------------------------------------------------------------------------------------------------------

Rng& World::random()
{
    checkNotParallel("random");
    return rng;
}

void World::runStep()
{
    // Timed by hand: an aborted step's time must not reach the next sample (a scope timer would add it after the
    // discard)
    const auto stepStart = prof.enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    recordsAtStepStart   = log.appendedTotal();
    // Queued player actions become ActionEvents; with the events of the last step they become readable
    deliverActions();
    events.beginStep();
    stepEventsBegun = true;
    try
    {
        for (int p = kFirstPhase; p <= kLastPhase; ++p)
        {
            phase = Phase(p);
            Profiler::Timer phaseTimer(prof, phaseScopes[size_t(p)]);
            // Teleports made in phase 7 reach the children before extract reads them (§7.4)
            if (phase == Phase::Extract) propagateTeleportsBeforeExtract(*this);
            // Received messages before the phase's systems (§8.1 phase 1); a client applies the newest snapshot
            if (phase == Phase::Input && net)
            {
                receiveMessages();
                repl->afterReceive(false);
            }
            const auto& list = systems.systems(phase);
            const RunOn skip = isHost() ? RunOn::Client : RunOn::Host;
            for (size_t i = 0; i < list.size(); ++i)
            {
                // RunOn filtering (§8.2): the host runs Host and Both, a client Client and Both
                if (list[i].runOn == skip) continue;
                // ownerOf/ownedBy must not rebuild the cache inside the system's parallel loops (R-PAR-1)
                if (ownerCacheDirty) rebuildOwnerCache();
                Profiler::Timer systemTimer(prof, systemScopes[size_t(p)][i]);
                list[i].fn(*this, stepDt);
            }
            {
                Profiler::Timer flushTimer(prof, flushScopes[size_t(p)]);
                prof.addCounter(commandsCounter, double(commands.size()));
                flushCommands();
            }
            // After phase 8's systems and their commands: replication (host), owner state (owners), interpolation
            if (phase == Phase::Extract)
            {
                repl->extract(false);
                // The render extract (S19b): after the interpolation, its own scope
                Profiler::Timer extractTimer(prof, extractScope);
                extractor->run();
            }
        }
        // Children whose parent was teleported after their last update (phase 8) follow in the next phase 6
        collectLateChildTeleports(*this);
        // The step's messages go out after phase 8
        flushOutbox();
    }
    catch (...)
    {
        // The world stays usable: the step is incomplete (the changes of the finished phases remain), the pending
        // commands are dropped and their reserved entities released; the step index does not advance
        abortStep();
        prof.discardSample();
        throw;
    }
    phase              = Phase::None;
    stepEventsBegun    = false;
    peerListsCommitted = true;
    // The step completed: the staged replication bookkeeping is committed
    repl->commit();
    stepChanged.clear();
    reg.shiftTransformFlags(trackStepChanges > 0 ? &stepChanged : nullptr);
    // The extract's staged pairs become current (an aborted step does not shift them; S19b)
    extractor->commit();
    // The records of this step (consumers that read after the step, e.g. the world API, keep older ones a step longer)
    prof.setCounter(recordsCounter, double(log.appendedTotal() - recordsAtStepStart));
    log.truncate();
    transitionsRequestedInStep.clear();
    steps++;
    prof.setCounter(entitiesCounter, double(reg.aliveCount()));
    prof.setCounter(objectsCounter, double(reg.registered(Behaviour::id) ? reg.poolById(Behaviour::id)->size() : 0));
    prof.setCounter(gridCounter, double(grid.size()));
    if (prof.enabled)
        prof.add(stepScope,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stepStart).count());
}

void World::stepNetworkOnly()
{
    SAIGA_ASSERT(!inStep(), "World::stepNetworkOnly inside a step");
    if (dispatchingStepListeners)
    {
        SAIGA_ASSERT(false, "World::stepNetworkOnly inside a step listener (not supported)");
        throw std::logic_error("World::stepNetworkOnly inside a step listener (not supported)");
    }
    try
    {
        phase = Phase::Input;
        if (net)
        {
            receiveMessages(true);
            repl->afterReceive(true);
        }
        flushCommands();
        if (isHost())
        {
            phase = Phase::Physics;
            repl->ownerPose(false);
            flushCommands();
        }
        // Derived child transforms follow (phase 6's hierarchy)
        phase = Phase::Transform;
        transformSystem(*this, stepDt);
        flushCommands();
        phase = Phase::Extract;
        repl->extract(true);
        flushCommands();
        flushOutbox();
    }
    catch (...)
    {
        abortStep();
        throw;
    }
    phase              = Phase::None;
    peerListsCommitted = true;
    repl->commit();
}

void World::step(int n)
{
    SAIGA_ASSERT(!inStep(), "World::step inside a step");
    if (dispatchingStepListeners)
    {
        // Refused in every build: the listener dispatch is not re-entrant
        SAIGA_ASSERT(false, "World::step inside a step listener (not supported)");
        throw std::logic_error("World::step inside a step listener (not supported)");
    }
    for (int i = 0; i < n; ++i)
    {
        runStep();
        // After runStep: its step timer has stopped
        prof.endSample();
        runStepListeners();
    }
}

void World::advance(double frameDt)
{
    SAIGA_ASSERT(!inStep(), "World::advance inside a step");
    if (dispatchingStepListeners)
    {
        SAIGA_ASSERT(false, "World::advance inside a step listener (not supported)");
        throw std::logic_error("World::advance inside a step listener (not supported)");
    }
    if (paused) return;
    // NaN would poison the accumulator for good; +inf is a very long frame (the backlog is dropped below)
    if (std::isnan(frameDt)) frameDt = 0;
    if (std::isinf(frameDt)) frameDt = frameDt > 0 ? stepDt * (config.maxStepsPerAdvance + 1) : 0;
    accumulator += std::max(frameDt, 0.0) * scale;
    int n = 0;
    while (accumulator >= stepDt && n < config.maxStepsPerAdvance)
    {
        {
            runStep();
            // After runStep: its step timer has stopped
            prof.endSample();
            runStepListeners();
        }
        accumulator -= stepDt;
        n++;
    }
    if (accumulator >= stepDt)
    {
        // The backlog beyond maxStepsPerAdvance is dropped (a slow frame does not cause a spiral of catch-up steps)
        // The whole steps that are discarded (0.999 dt of the backlog is kept)
        const double kept      = 0.999 * stepDt;
        const double discarded = std::floor((accumulator - kept) / stepDt);
        // Clamped before the conversion (a huge or infinite backlog)
        dropped += discarded >= 1e18 ? uint64_t(1e18) : uint64_t(discarded);
        accumulator = kept;
    }
}

int World::stepsForAdvance(double frameDt)
{
    // The guards of advance (local review: the caller steps right after it)
    SAIGA_ASSERT(!inStep(), "World::stepsForAdvance inside a step");
    if (dispatchingStepListeners)
    {
        SAIGA_ASSERT(false, "World::stepsForAdvance inside a step listener (not supported)");
        throw std::logic_error("World::stepsForAdvance inside a step listener (not supported)");
    }
    if (paused) return 0;
    // As advance: NaN is ignored, +inf is a long frame
    if (std::isnan(frameDt)) frameDt = 0;
    if (std::isinf(frameDt)) frameDt = frameDt > 0 ? stepDt * (config.maxStepsPerAdvance + 1) : 0;
    accumulator += std::max(frameDt, 0.0) * scale;
    int n = 0;
    while (accumulator >= stepDt && n < config.maxStepsPerAdvance)
    {
        accumulator -= stepDt;
        n++;
    }
    if (accumulator >= stepDt)
    {
        const double kept      = 0.999 * stepDt;
        const double discarded = std::floor((accumulator - kept) / stepDt);
        dropped += discarded >= 1e18 ? uint64_t(1e18) : uint64_t(discarded);
        accumulator = kept;
    }
    return n;
}

void World::setTimeScale(double s)
{
    SAIGA_ASSERT(std::isfinite(s) && s >= 0, "World::setTimeScale: the scale must be finite and not negative");
    scale = s;
}

void World::setWorkerThreads(int n)
{
    SAIGA_ASSERT(n >= 1, "World::setWorkerThreads: at least 1 (the calling thread)");
    SAIGA_ASSERT(!inStep(), "World::setWorkerThreads inside a step");
    pool.setWorkerThreads(n - 1);
}

uint64_t World::hash() const
{
    SAIGA_ASSERT(!inStep() && commands.empty(), "World::hash inside a step");
    Hasher h;
    reg.hashInto(h);
    const Rng::State s = rng.state();
    h.addU64(s.state).addU64(s.increment);
    events.hashPending(h);
    // The queued actions (input that becomes events) and the Persistent counter (S10)
    h.addU64(uint64_t(actions.size()));
    for (const QueuedAction& a : actions)
    {
        // The queue order (the sequence counter itself is not state: only the order matters)
        h.addU64(a.peer).addU32(a.action.player.index).addU32(a.action.player.generation);
        h.addU32(a.action.action);
        for (int i = 0; i < 3; ++i) h.addDouble(a.action.aim[i]);
    }
    h.addU64(persistentCounter);
    h.addU64(steps);
    return h.value();
}

// ---- network messages (S15)
// -----------------------------------------------------------------------------------------------------------

void World::enqueueMessage(uint8_t peer, const MessageList::Entry& entry, std::vector<uint8_t> payload)
{
    checkNotParallel("send");
    if (!net) SAIGA_EXIT_ERROR("World::send " + entry.name + ": the world has no transport");
    if (peer == net->localPeer()) SAIGA_EXIT_ERROR("World::send " + entry.name + ": to the own peer");
    if (entry.reliability == Reliability::Unreliable && payload.size() > kMaxMessagePayload)
        SAIGA_EXIT_ERROR("World::send: the unreliable message " + entry.name + " has " +
                         std::to_string(payload.size()) + " bytes (at most " + std::to_string(kMaxMessagePayload) +
                         ")");
    if (payload.size() > kMaxReliablePayload)
        SAIGA_EXIT_ERROR("World::send: the message " + entry.name + " has " + std::to_string(payload.size()) +
                         " bytes (at most " + std::to_string(kMaxReliablePayload) + ")");
    if (loadHookPhase != 0)
    {
        // configure / onLoad rebuild local state from the save; telling the peers is the load listeners' job
        netCounts.droppedDuringObjectStart++;
        SAIGA_LOG("net", "send of %s during a load's object start dropped", entry.name.c_str());
        SAIGA_ASSERT(false, "World::send during configure / onLoad of a load (dropped)");
        return;
    }
    outbox.push_back(
        OutboxEntry{peer, entry.channel, entry.reliability, phase, sendSequence[peer]++, entry.id, std::move(payload)});
    if (!inStep() && sendBatchDepth == 0) flushOutbox();
}

void World::sendRaw(uint8_t peer, MessageId id, const uint8_t* data, size_t size)
{
    const MessageList::Entry* e = messages.find(id);
    if (!e || !e->raw) SAIGA_EXIT_ERROR("World::sendRaw: the message " + std::to_string(id) + " is not a raw message");
    // The size before the buffer is read
    if (e->reliability == Reliability::Unreliable ? size > kMaxMessagePayload : size > kMaxReliablePayload)
        SAIGA_EXIT_ERROR("World::sendRaw: the message " + e->name + " has " + std::to_string(size) + " bytes");
    enqueueMessage(peer, *e, std::vector<uint8_t>(data, data + size));
}

void World::endSendBatch()
{
    SAIGA_ASSERT(sendBatchDepth > 0, "endSendBatch without beginSendBatch");
    if (--sendBatchDepth == 0 && !inStep()) flushOutbox();
}

void World::flushOutbox()
{
    if (outbox.empty() || !net) return;
    // Groups per (peer, channel, reliability) in first-seen order; send order inside a group
    struct Group
    {
        uint8_t peer;
        Channel channel;
        Reliability reliability;
        std::vector<size_t> entries;
    };
    std::vector<Group> groups;
    for (size_t i = 0; i < outbox.size(); ++i)
    {
        const OutboxEntry& e = outbox[i];
        Group* g             = nullptr;
        for (Group& x : groups)
            if (x.peer == e.peer && x.channel == e.channel && x.reliability == e.reliability) g = &x;
        if (!g)
        {
            groups.push_back(Group{e.peer, e.channel, e.reliability, {}});
            g = &groups.back();
        }
        g->entries.push_back(i);
    }
    uint64_t bytes = 0;
    for (const Group& g : groups)
    {
        size_t k = 0;
        while (k < g.entries.size())
        {
            // [base sequence][count] then per message [sequence delta][id u16][length][payload]: the messages of
            // the packet are chosen first, then written straight into it (one copy of each payload)
            const uint64_t base = outbox[g.entries[k]].sequence;
            uint64_t previous   = base;
            size_t end = k, bodySize = 0;
            while (end < g.entries.size())
            {
                const OutboxEntry& e = outbox[g.entries[end]];
                const size_t size =
                    varintSize(e.sequence - previous) + 2 + varintSize(e.payload.size()) + e.payload.size();
                // A packet holds what fits kMaxPacket; a single larger (reliable) message is a packet of its own
                if (end > k && kMaxPacketHeader + bodySize + size > kMaxPacket) break;
                bodySize += size;
                previous = e.sequence;
                end++;
            }
            std::vector<uint8_t> packet;
            packet.reserve(varintSize(base) + varintSize(end - k) + bodySize);
            writeVarint(packet, base);
            writeVarint(packet, end - k);
            previous = base;
            for (; k < end; ++k)
            {
                const OutboxEntry& e = outbox[g.entries[k]];
                writeVarint(packet, e.sequence - previous);
                packet.push_back(uint8_t(e.id & 0xFF));
                packet.push_back(uint8_t(e.id >> 8));
                writeVarint(packet, e.payload.size());
                packet.insert(packet.end(), e.payload.begin(), e.payload.end());
                previous = e.sequence;
            }
            bytes += packet.size();
            net->send(g.peer, g.channel, packet, g.reliability);
        }
    }
    netCounts.messagesOut += outbox.size();
    netCounts.bytesOut += bytes;
    if (counterMessagesOut >= 0)
    {
        prof.addCounter(counterMessagesOut, double(outbox.size()));
        prof.addCounter(counterSentBytes, double(bytes));
    }
    outbox.clear();
}

void World::discardOutboxAfterLoad()
{
    netCounts.droppedOnLoad += outbox.size();
    outbox.clear();
}

void World::rejectInput(uint8_t peer, const std::string& why)
{
    netCounts.rejected++;
    if (counterRejected >= 0) prof.addCounter(counterRejected, 1);
    // At most 8 lines per peer and step (input from outside the process must not flood the log)
    if (rejectLogs[peer]++ < 8) SAIGA_LOG("net", "rejected input from peer %d: %s", int(peer), why.c_str());
}

bool World::decodePacket(uint8_t peer, const std::vector<uint8_t>& packet, std::vector<ReceivedMessage>& out)
{
    const uint8_t* d = packet.data();
    const size_t n   = packet.size();
    size_t pos       = 0;
    uint64_t base, count;
    if (!readVarint(d, n, pos, base) || !readVarint(d, n, pos, count) || count == 0 || count > (n - pos) / 4)
    {
        rejectInput(peer, "a malformed packet header");
        return false;
    }
    const size_t first = out.size();
    uint64_t sequence  = base;
    for (uint64_t i = 0; i < count; ++i)
    {
        uint64_t delta, length;
        // The first message has the base sequence, each later one a delta of at least 1 (send order, gaps allowed)
        if (!readVarint(d, n, pos, delta) || (i == 0 ? delta != 0 : delta == 0 || sequence + delta < sequence) ||
            pos + 2 > n)
        {
            out.resize(first);
            rejectInput(peer, "a malformed message header");
            return false;
        }
        sequence += delta;
        const MessageId id = MessageId(d[pos] | (uint16_t(d[pos + 1]) << 8));
        pos += 2;
        if (!readVarint(d, n, pos, length) || length > n - pos)
        {
            out.resize(first);
            rejectInput(peer, "a message length beyond the packet");
            return false;
        }
        out.push_back(ReceivedMessage{peer, sequence, id, std::vector<uint8_t>(d + pos, d + pos + length)});
        pos += size_t(length);
    }
    if (pos != n)
    {
        out.resize(first);
        rejectInput(peer, "bytes after the last message");
        return false;
    }
    return true;
}

void World::receiveMessages(bool engineOnly)
{
    // The peer lists of a completed step are replaced; after an aborted step they stay until a step completes
    if (peerListsCommitted)
    {
        joined.clear();
        left.clear();
        peerListsCommitted = false;
    }
    rejectLogs.fill(0);
    std::vector<ReceivedMessage> received;
    uint64_t bytes = 0;
    TransportEvent ev;
    while (net->poll(ev))
    {
        if (ev.type == TransportEvent::Connected)
        {
            if (std::find(peers.begin(), peers.end(), ev.peer) == peers.end()) peers.push_back(ev.peer);
            joined.push_back(ev.peer);
        }
        else if (ev.type == TransportEvent::Disconnected)
        {
            peers.erase(std::remove(peers.begin(), peers.end(), ev.peer), peers.end());
            left.push_back(ev.peer);
        }
        else
        {
            bytes += ev.packet.size();
            decodePacket(ev.peer, ev.packet, received);
        }
    }
    netCounts.bytesIn += bytes;
    if (counterReceivedBytes >= 0) prof.addCounter(counterReceivedBytes, double(bytes));
    // Within a round the order is (peer, sequence), independent of the arrival order; a leftover of an aborted step
    // comes first
    std::stable_sort(received.begin(), received.end(), [](const ReceivedMessage& a, const ReceivedMessage& b)
                     { return a.peer != b.peer ? a.peer < b.peer : a.sequence < b.sequence; });
    for (auto& m : received) inbox.push_back(std::move(m));
    size_t i = 0;
    // stepNetworkOnly: game messages stay queued (in order) for the next step
    std::vector<ReceivedMessage> deferred;
    try
    {
        for (; i < inbox.size(); ++i)
        {
            const ReceivedMessage& m = inbox[i];
            if (engineOnly && m.id >= MessageId(EngineMessage::FirstGame))
            {
                deferred.push_back(std::move(inbox[i]));
                continue;
            }
            const MessageList::Entry* e = messages.find(m.id);
            if (!e)
            {
                rejectInput(m.peer, "an unknown message " + std::to_string(m.id));
                continue;
            }
            std::string error;
            const size_t outboxMark = outbox.size();
            if (!e->dispatch(*this, m.peer, m.payload.data(), m.payload.size(), error))
            {
                rejectInput(m.peer, e->name + ": " + error);
                continue;
            }
            netCounts.messagesIn++;
            if (counterMessagesIn >= 0) prof.addCounter(counterMessagesIn, 1);
            // The handler's structural changes apply before the next handler: a message is consumed together with them
            // and with its sends (an abort later in phase 1 must not drop a consumed message's reply)
            flushCommands();
            for (size_t k = outboxMark; k < outbox.size(); ++k) outbox[k].phase = Phase::None;
        }
    }
    catch (...)
    {
        // The throwing message is consumed too (a poison message must not abort every later step); its partial
        // commands are dropped by abortStep; the rest is handled first in the next step
        netCounts.handlerExceptions++;
        inbox.erase(inbox.begin(), inbox.begin() + std::ptrdiff_t(i + 1));
        inbox.insert(inbox.begin(), std::make_move_iterator(deferred.begin()), std::make_move_iterator(deferred.end()));
        throw;
    }
    inbox.swap(deferred);
}

}  // namespace Engine
}  // namespace Saiga

namespace Saiga
{
namespace Engine
{
// ---- render assets and the render sink (S19b)
// ----------------------------------------------------------------------------------------------

Vulkan::MeshId World::registerMesh(const std::string& name, std::shared_ptr<const Vulkan::MeshLods> lods,
                                   Vulkan::MaterialId material)
{
    SAIGA_ASSERT(!inStep(), "World::registerMesh inside a step");
    if (name.empty()) SAIGA_EXIT_ERROR("World::registerMesh: an empty name");
    if (meshByName.count(name)) SAIGA_EXIT_ERROR("World::registerMesh: the mesh \"" + name + "\" is registered twice");
    if (!lods || lods->lods.empty()) SAIGA_EXIT_ERROR("World::registerMesh: the mesh \"" + name + "\" has no LODs");
    if (material < Vulkan::kGameReservedIds || material - Vulkan::kGameReservedIds >= materialTable.size())
        SAIGA_EXIT_ERROR("World::registerMesh: the mesh \"" + name + "\" names an unregistered material");
    if (Vulkan::kGameReservedIds + meshTable.size() > 0xFFFF) SAIGA_EXIT_ERROR("World::registerMesh: too many meshes");
    const Vulkan::MeshId id = Vulkan::MeshId(Vulkan::kGameReservedIds + meshTable.size());
    meshTable.push_back(MeshEntry{name, std::move(lods), material});
    meshByName[name] = id;
    if (RenderSink* sink = extractor->sink())
    {
        sink->registerMesh(id, name, meshTable.back().lods, material);
        extractor->meshRegistered();
    }
    return id;
}

Vulkan::MaterialId World::registerMaterial(const std::string& name, const Vulkan::MaterialData& data)
{
    SAIGA_ASSERT(!inStep(), "World::registerMaterial inside a step");
    if (name.empty()) SAIGA_EXIT_ERROR("World::registerMaterial: an empty name");
    if (materialByName.count(name))
        SAIGA_EXIT_ERROR("World::registerMaterial: the material \"" + name + "\" is registered twice");
    if (Vulkan::kGameReservedIds + materialTable.size() > 0xFFFF)
        SAIGA_EXIT_ERROR("World::registerMaterial: too many materials");
    const Vulkan::MaterialId id = Vulkan::MaterialId(Vulkan::kGameReservedIds + materialTable.size());
    materialTable.push_back(MaterialEntry{name, data});
    materialByName[name] = id;
    if (RenderSink* sink = extractor->sink()) sink->registerMaterial(id, name, data);
    return id;
}

void World::registerMeshAlias(const std::string& oldName, const std::string& name)
{
    SAIGA_ASSERT(!inStep(), "World::registerMeshAlias inside a step");
    if (!meshByName.count(name))
        SAIGA_EXIT_ERROR("World::registerMeshAlias: \"" + name + "\" is not a registered mesh");
    if (meshByName.count(oldName))
        SAIGA_EXIT_ERROR("World::registerMeshAlias: \"" + oldName + "\" is a registered mesh");
    for (const auto& a : meshAliases)
        if (a.first == oldName) SAIGA_EXIT_ERROR("World::registerMeshAlias: \"" + oldName + "\" has an alias already");
    meshAliases.push_back({oldName, name});
    // Entities that name oldName (kept unresolved by a load or the add path) are drawn now
    if (extractor->sink()) extractor->meshRegistered();
}

Vulkan::MeshId World::findMesh(const std::string& name) const
{
    auto it = meshByName.find(name);
    if (it != meshByName.end()) return it->second;
    for (const auto& a : meshAliases)
        if (a.first == name) return findMesh(a.second);
    return 0;
}

Vulkan::MaterialId World::findMaterial(const std::string& name) const
{
    auto it = materialByName.find(name);
    return it == materialByName.end() ? Vulkan::MaterialId(0) : it->second;
}

std::string World::meshName(Vulkan::MeshId id) const
{
    if (id < Vulkan::kGameReservedIds || id - Vulkan::kGameReservedIds >= meshTable.size()) return {};
    return meshTable[id - Vulkan::kGameReservedIds].name;
}

uint64_t World::renderTableHash() const
{
    Hasher h;
    h.addU64(meshTable.size());
    for (size_t i = 0; i < meshTable.size(); ++i)
        h.addU32(uint32_t(Vulkan::kGameReservedIds + i)).addString(meshTable[i].name).addU32(meshTable[i].material);
    h.addU64(materialTable.size());
    for (size_t i = 0; i < materialTable.size(); ++i)
        h.addU32(uint32_t(Vulkan::kGameReservedIds + i)).addString(materialTable[i].name);
    h.addU64(meshAliases.size());
    for (const auto& a : meshAliases) h.addString(a.first).addString(a.second);
    return h.value();
}

void World::setRenderSink(RenderSink* sink)
{
    SAIGA_ASSERT(!inStep(), "World::setRenderSink inside a step");
    if (sink)
    {
        // Every registration so far (materials first: meshes name them)
        for (size_t i = 0; i < materialTable.size(); ++i)
            sink->registerMaterial(Vulkan::MaterialId(Vulkan::kGameReservedIds + i), materialTable[i].name,
                                   materialTable[i].data);
        for (size_t i = 0; i < meshTable.size(); ++i)
            sink->registerMesh(Vulkan::MeshId(Vulkan::kGameReservedIds + i), meshTable[i].name, meshTable[i].lods,
                               meshTable[i].material);
        // The render settings per space (S21): one call per SpaceId (from S23a several spaces may share a template)
        for (size_t i = 0; i < spaceCount(); ++i)
        {
            const SpaceTemplate& t = spaceTemplate(SpaceId(i));
            Vulkan::SpaceRenderSettings settings;
            settings.openWorld  = t.kind == SpaceKind::OpenWorld;
            settings.background = t.background.cast<float>();
            sink->setSpaceSettings(SpaceId(i), settings);
        }
    }
    extractor->setSink(sink);
    // The terrain's send point (S20): a new sink receives what the store holds
    terrainSys->setSink(sink);
}

RenderSink* World::renderSink() const
{
    return extractor->sink();
}

bool World::resolveRenderMesh(std::string& name, Vulkan::MeshId& id) const
{
    id = 0;
    if (name.empty()) return true;
    auto it = meshByName.find(name);
    if (it != meshByName.end())
    {
        id = it->second;
        return true;
    }
    for (const auto& a : meshAliases)
        if (a.first == name)
        {
            std::string renamed = a.second;
            if (!resolveRenderMesh(renamed, id)) return false;
            name = renamed;
            return true;
        }
    return false;
}

void World::validateRenderInstance(Entity e, RenderInstance& value) const
{
    Vulkan::MeshId id = 0;
    if (!resolveRenderMesh(value.mesh, id))
    {
        // Logged in every build (once per write); the name is kept: tracked, not drawn until a mesh of that name is
        // registered (World::registerMesh creates the instance then). A load still rejects an unknown name
        SAIGA_LOG_E("render", e,
                    "RenderInstance: the mesh \"%s\" is not registered (tracked, drawn once it is registered)",
                    value.mesh.c_str());
        id = 0;
    }
    value.cachedMesh = id;
}

void World::renderMeshWritten(Entity e, bool added, const std::string* previous)
{
    RenderInstance& ri = reg.get<RenderInstance>(e);
    validateRenderInstance(e, ri);
    if (added || (previous && *previous == ri.mesh)) return;
    // The name record: the extract recreates the instance (replication presence and changes_since report it so)
    auto record = log.prepare(2);
    record.append(e, RenderInstance::id, false);
    record.append(e, RenderInstance::id, true);
}

void World::setRenderMesh(Entity e, const std::string& name)
{
    checkNotParallel("setRenderMesh");
    std::string resolved = name;
    Vulkan::MeshId id    = 0;
    if (!resolveRenderMesh(resolved, id))
    {
        SAIGA_ASSERT(false, "World::setRenderMesh: the mesh \"" + name + "\" is not registered");
        return;
    }
    if (deferring() || (flushing && reg.reserved(e)))
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), "setRenderMesh: the entity is not alive");
        record(makeCommand([e, resolved](World& w) { w.setRenderMesh(e, resolved); }));
        return;
    }
    if (flushing && !reg.alive(e)) return;
    SAIGA_ASSERT(reg.alive(e), "setRenderMesh: the entity is not alive");
    if (!reg.has<RenderInstance>(e))
    {
        addNow<RenderInstance>(e, RenderInstance(resolved));
        return;
    }
    const std::string previous = reg.read<RenderInstance>(e).mesh;
    // A load hook must not change saved state (the add branch is checked by addNow); the same name is no change
    if (loadHookPhase && previous != resolved && !checkLoadHook("setRenderMesh", e, RenderInstance::id)) return;
    reg.get<RenderInstance>(e).mesh = resolved;
    renderMeshWritten(e, false, &previous);
}

}  // namespace Engine
}  // namespace Saiga
