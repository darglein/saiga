/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/JobPool.h"
#include "saiga/engine/core/Profiler.h"
#include "saiga/engine/core/Rng.h"
#include "saiga/engine/data/DataRef.h"
#include "saiga/engine/data/DataStore.h"
#include "saiga/engine/ecs/Registry.h"
#include "saiga/engine/gameplay/ActionList.h"
#include "saiga/engine/gameplay/ClassList.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/gameplay/TickPolicy.h"
#include "saiga/engine/net/MessageList.h"
#include "saiga/engine/net/NetIds.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/net/Transport.h"
#include "saiga/engine/save/LoadResult.h"
#include "saiga/engine/save/MigrationList.h"
#include "saiga/engine/world/Commands.h"
#include "saiga/engine/world/EventQueues.h"
#include "saiga/engine/world/EventRegistry.h"
#include "saiga/engine/world/Phase.h"
#include "saiga/engine/world/SpatialGrid.h"
#include "saiga/engine/world/StructuralLog.h"
#include "saiga/engine/world/SystemList.h"
#include "saiga/export.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <type_traits>
#include <unordered_map>

namespace Saiga
{
namespace Engine
{
// Game policies of a world (R-MECH-1; docs/engine_design.md §7.9). Later stages add their fields.
struct WorldConfig
{
    double stepsPerSecond  = 60;
    int maxStepsPerAdvance = 4;
    TickPolicy ticking;  // §8.4 (S10)
    // RenderPose samples the interpolation buffer this many steps behind the (estimated) host step (§13.4, S16)
    int interpolationDelaySteps = 6;
};

struct WorldSetup
{
    uint64_t seed = 1;
    // Threads of the frame JobPool for parallelEach, including the calling thread (1: serial)
    int workerThreads = 1;
    WorldConfig config;
    // The game's registration lists; the engine's lists are always registered first
    void (*components)(Registry&)  = nullptr;
    void (*events)(EventRegistry&) = nullptr;
    void (*systems)(SystemList&)   = nullptr;
    void (*classes)(ClassList&)    = nullptr;  // game object classes (S10)
    void (*actions)(ActionList&)   = nullptr;  // player actions and held buttons by name (S13)
    // Network messages (§13.6, S15): the engine's first, then the game's (IDs from 64)
    void (*messages)(MessageList&) = nullptr;
    // The transport to the other peers (not owned; it outlives the world's use of it); null: single player
    Transport* transport = nullptr;
    // Host (default; single player is a host without peers) or Client (S16, §13.1); localPeer must equal
    // transport->localPeer() when a transport is set (the host is peer 0)
    NetRole role      = NetRole::Host;
    uint8_t localPeer = 0;
    // Migrations of changed and removed saved types (§15.2, S11)
    void (*migrations)(MigrationList&) = nullptr;
    // Data files (§14, S11): one JSON file per registered type in dataDirectory, loaded by the constructor (an error
    // stops the program in every build, R-DATA-1). sharedData: an already loaded store of the same types (worlds of
    // one setup, e.g. scenario runs) instead of reading the files
    void (*dataTypes)(DataTypeList&) = nullptr;
    std::string dataDirectory;
    std::shared_ptr<DataStore> sharedData;
    // The shared store must not be reloaded by this world (reloadData and the reload_data operation fail): the worlds
    // run_scenario creates share the live world's store (S14)
    bool sharedDataReadOnly = false;
    // Times every step, phase, system and command flush and counts entities, objects and updates (Profiler, S10);
    // false: no clock calls
    bool profiling = true;

    // Constant configuration that is not a data file (a level, a track, a height field; §7.9, S13b): built once per
    // process before the worlds, shared by every world of this setup (scenario runs too), read in configure() and in
    // systems through World::context<T>(). Not state: it is not saved and must not change after a world exists.
    template <class T>
    void setContext(std::shared_ptr<const T> c)
    {
        context     = std::move(c);
        contextType = context ? &typeid(T) : nullptr;
    }
    // A non-const pointer (std::make_shared<Level>()): stored as const
    template <class T>
    void setContext(std::shared_ptr<T> c)
    {
        setContext(std::shared_ptr<const T>(std::move(c)));
    }
    std::shared_ptr<const void> context;
    const std::type_info* contextType = nullptr;
};

/**
 * The simulation (docs/engine_design.md §7.6-§7.10, §8.1-§8.3, §8.6). Owns the Registry and runs fixed steps through
 * the phases of §8.1; the systems of a phase run in registration order and the commands they record are applied after
 * the phase.
 *
 * Structural changes (create, destroy, add, remove):
 * - between steps (setup code, tests, loading, tool handlers) and while commands are applied: immediately;
 * - during a step (systems): as commands, applied at the end of the phase in recording order. create() reserves the
 *   entity and returns it at once; views skip it and get/read/has assert until the command is applied; add on it is
 *   allowed (also a command). Commands on an entity that was destroyed earlier in the same flush are dropped.
 * Every applied change is appended to the structural log. Destroying a parent destroys its descendants (§7.4).
 */
class Replication;

class SAIGA_ENGINE_API World
{
   public:
    explicit World(const WorldSetup& setup);
    ~World();
    World(const World&)            = delete;
    World& operator=(const World&) = delete;

    // ---- entities and components (§7.10) -------------------------------------------------------------------------

    Entity create();
    // A game object of class T at position (space 0; spawnIn: S21 adds more spaces): Transform, Behaviour and the
    // initial components, then configure() and onSpawn() (§9.2). During a step the entity is reserved and returned at
    // once, the rest happens at the flush; between steps at once. Defined in gameplay/Behaviour.h.
    template <class T, class... C>
    Entity spawn(const Vec3& position, C&&... initial);
    template <class T, class... C>
    Entity spawnIn(SpaceId space, const Vec3& position, C&&... initial);
    // The game object's onDestroy() runs first (children before parents in the cascade). World-state entities cannot
    // be destroyed.
    void destroy(Entity e);
    bool alive(Entity e) const { return reg.alive(e); }

    template <class T, class... A>
    void add(Entity e, A&&... args);
    template <class T>
    void remove(Entity e);
    // Parent is read-only through the World: change it with remove + add (the hierarchy order and the cycle check
    // depend on it)
    template <class T>
    T& get(Entity e)
    {
        static_assert(T::id != ComponentId(EngineComponent::Parent), "Parent: use remove<Parent> and add<Parent>");
        static_assert(T::id != ComponentId(EngineComponent::Persistent), "Persistent is read-only (makePersistent)");
        static_assert(T::id != ComponentId(EngineComponent::NetOwner), "NetOwner is read-only (setOwner)");
        return reg.get<T>(e);
    }
    template <class T>
    T* tryGet(Entity e)
    {
        static_assert(T::id != ComponentId(EngineComponent::Parent), "Parent: use remove<Parent> and add<Parent>");
        static_assert(T::id != ComponentId(EngineComponent::Persistent), "Persistent is read-only (makePersistent)");
        static_assert(T::id != ComponentId(EngineComponent::NetOwner), "NetOwner is read-only (setOwner)");
        return reg.tryGet<T>(e);
    }
    template <class T>
    const T& read(Entity e) const
    {
        return reg.read<T>(e);
    }
    template <class T>
    const T* tryRead(Entity e) const
    {
        return reg.tryRead<T>(e);
    }
    template <class T>
    bool has(Entity e) const
    {
        return reg.has<T>(e);
    }
    // In phases 2-5 only Simulated entities (§12.3); viewAll: all alive entities
    template <class... C>
    View<C...> view()
    {
        static_assert(parentReadOnly<C...>(),
                      "Parent, Persistent and NetOwner in a view must be const (Parent: remove and add; Persistent: "
                      "makePersistent; NetOwner: setOwner)");
        return reg.view<C...>(simulatedOnly(phase) ? Registry::kSimulated : 0);
    }
    template <class... C>
    View<C...> viewAll()
    {
        static_assert(parentReadOnly<C...>(),
                      "Parent, Persistent and NetOwner in a view must be const (Parent: remove and add; Persistent: "
                      "makePersistent; NetOwner: setOwner)");
        return reg.view<C...>(0);
    }
    template <class... C>
    static ComponentMask mask()
    {
        return Registry::mask<C...>();
    }

    EventQueues events;

    // ---- gameplay (§7.8, §9.3, §9.6) ------------------------------------------------------------------------------

    // A one-shot player action: delivered on the host as ActionEvent, readable in the next step, in (peer, sequence)
    // order, exactly once. The player must be alive and have a Player component. On a client the player must be owned
    // by this peer: the action is sent to the host as a PlayerAction message (S16).
    void queueAction(const PlayerAction& action);

    // ---- network messages (§13.6; S15) ---------------------------------------------------------------------------

    // Cumulative counts of the world's message traffic (not reset by aborted steps)
    struct NetStats
    {
        uint64_t messagesIn = 0, messagesOut = 0, bytesIn = 0, bytesOut = 0;
        uint64_t rejected                 = 0;  // malformed packets, unknown IDs, payloads that do not decode
        uint64_t handlerExceptions        = 0;
        uint64_t droppedOnLoad            = 0;  // outbox entries a load discarded
        uint64_t droppedDuringObjectStart = 0;  // sends from configure / onLoad of a load
    };
    // A registered message to a peer: encoded at once into the outbox (flushed after phase 8, or at once between steps
    // outside a batching scope). An unregistered type, no transport or the own peer stop the program in every build.
    template <class M>
    void send(uint8_t peer, const M& message)
    {
        const MessageList::Entry* e = messages.find(M::id);
        if (!e || e->raw || e->name != M::name)
            SAIGA_EXIT_ERROR(std::string("World::send: the message ") + M::name + " is not registered");
        std::vector<uint8_t> payload;
        std::string error;
        if (!toBytes(message, payload, &error)) SAIGA_EXIT_ERROR(std::string("World::send ") + M::name + ": " + error);
        enqueueMessage(peer, *e, std::move(payload));
    }
    // A raw message (MessageList::addRaw): the size is checked before the buffer is read
    void sendRaw(uint8_t peer, MessageId id, const uint8_t* data, size_t size);
    // Sends the outbox now (coalesced into packets per peer, channel and reliability)
    void flushOutbox();
    // Sends between steps are batched while a scope is open (world API calls, step listener dispatch, loads) and
    // flushed when the outermost closes
    void beginSendBatch() { sendBatchDepth++; }
    void endSendBatch();
    Transport* transport() const { return net; }
    const MessageList& messageList() const { return messages; }
    // The connected peers; the peers that connected / disconnected in the last receive (kept until a step completes)
    const std::vector<uint8_t>& connectedPeers() const { return peers; }
    const std::vector<uint8_t>& peersJoined() const { return joined; }
    const std::vector<uint8_t>& peersLeft() const { return left; }
    const NetStats& netStats() const { return netCounts; }
    // The role of this world (WorldSetup::role, localPeer; S16)
    uint8_t localPeer() const { return selfPeer; }
    bool isHost() const { return role == NetRole::Host; }
    NetRole netRole() const { return role; }
    // The entities whose root is owned by peer (R-NET-2, §13.3): NetOwner of the root, peer 0 (the host) without one.
    // One lookup per entity in a cached owner table (rebuilt after a Parent or NetOwner change and after a load). As
    // view() otherwise (Simulated in phases 2-5).
    template <class... C>
    View<C...> ownedBy(uint8_t peer)
    {
        static_assert(parentReadOnly<C...>(),
                      "Parent, Persistent and NetOwner in a view must be const (remove and add, makePersistent, "
                      "setOwner)");
        if (ownerCacheDirty) rebuildOwnerCache();
        return reg.ownedView<C...>(simulatedOnly(phase) ? Registry::kSimulated : 0, ownerCache, peer);
    }
    // The owner of the entity's root (0: the host)
    uint8_t ownerOf(Entity e) const
    {
        if (ownerCacheDirty) rebuildOwnerCache();
        return e.index < ownerCache.size() ? ownerCache[e.index] : uint8_t(0);
    }
    // Host: the owner of a replicated root (and its subtree), before the entity's spawn was sent (asserted); no
    // transfer later (§13.3). A Player's Player.peer must equal peer. Peer 0 is the host.
    void setOwner(Entity root, uint8_t peer);
    // Host: a teleport of a root. Before its spawn or for a host-owned entity a plain Transform::teleport; for a
    // spawned client-owned entity the pose is applied here too and sent to the owner as an OwnerCommand with the
    // incremented command counter (owner states with an older counter are dropped, §13.3)
    void ownerTeleport(Entity root, const Vec3& position, const Quat& rotation);
    // Engine messages only (receive, owner poses without extrapolation, phase 6's transform system, replication and
    // owner state, the flush): no game systems, listeners, actions or events; the step index and the transform flags
    // do not change (end-of-scenario sync, §17.3)
    void stepNetworkOnly();
    // Replication state (engine-internal; S16)
    Replication& replication() { return *repl; }
    const Replication& replication() const { return *repl; }
    // World-state components (registerGlobal) on the world-state entity, space-state components (registerSpaceState)
    // on the space-state entity of the space (space 0 before S21). A component that is not registered as global or
    // space state asserts.
    template <class T>
    T& global()
    {
        return reg.get<T>(globalEntity<T>());
    }
    template <class T>
    const T& readGlobal() const
    {
        return reg.read<T>(const_cast<World*>(this)->globalEntity<T>());
    }
    template <class T>
    T& spaceState(SpaceId space)
    {
        return reg.get<T>(spaceStateEntityOf<T>(space));
    }
    template <class T>
    const T& readSpaceState(SpaceId space) const
    {
        return reg.read<T>(const_cast<World*>(this)->spaceStateEntityOf<T>(space));
    }
    // The world-state entity (Entity{} if no global component is registered) and the space-state entity of a space
    Entity worldStateEntity() const { return worldState; }
    Entity spaceStateEntity(SpaceId space) const { return space < spaceStates.size() ? spaceStates[space] : Entity{}; }
    // A referable ID that survives saving (§9.6): the existing one, or the next value of the world's counter (the
    // Persistent component is added at once between steps, at the flush during a step)
    PersistentId makePersistent(Entity e);
    // The alive entity with this ID, Entity{} otherwise
    Entity resolve(PersistentId id) const;

    // ---- spatial queries (§9.5) ----------------------------------------------------------------------------------

    // The alive entities of the space within radius of center (Euclidean distance, boundary included) that have all
    // components of mask, sorted by entity index (deterministic). Positions as of the last phase 6: an entity moved
    // since is found at its old position, one created since is not found until then; destroyed entities are never
    // returned. Reserved entities are not found. Allowed in parallelEach (reads only).
    std::vector<Entity> queryRadius(SpaceId space, const Vec3& center, double radius,
                                    const ComponentMask& mask = ComponentMask()) const;
    // The same into out (cleared first; allocates only if out's capacity is smaller than the result)
    void queryRadius(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask,
                     std::vector<Entity>& out) const;

    // ---- time and steps (§8.3, §8.6) -----------------------------------------------------------------------------

    // The world's random stream (PCG32 seeded from WorldSetup::seed; part of the hash and saves). Not in parallelEach.
    Rng& random();
    double dt() const { return stepDt; }
    // The setup's context (WorldSetup::setContext); a missing context or another type stops the program in every build
    // (the cast would read another type)
    template <class T>
    const T& context() const
    {
        if (!contextPtr || !contextType || *contextType != typeid(T))
            SAIGA_EXIT_ERROR(std::string("World::context: no context of type ") + typeid(T).name() +
                             " (WorldSetup::setContext)");
        return *static_cast<const T*>(contextPtr.get());
    }
    bool hasContext() const { return contextPtr != nullptr; }
    // timeBase: a loaded save keeps its time also if it was saved with another step rate (S11)
    // On a client the estimated host time (§13.4, S16)
    double time() const { return timeBase + (double(steps - stepBase) + hostStepOffset) * stepDt; }
    uint64_t stepIndex() const { return steps; }
    // Interpolation factor between the last two steps, in [0, 1) (set by advance)
    double alpha() const { return accumulator / stepDt; }
    // accumulator += frameDt * timeScale; runs whole steps, at most maxStepsPerAdvance; the remaining backlog is
    // dropped (counted in droppedSteps). No steps while paused.
    void advance(double frameDt);
    // Runs exactly n steps (also while paused)
    void step(int n = 1);
    void setPaused(bool p) { paused = p; }
    bool isPaused() const { return paused; }
    void setTimeScale(double s);
    double timeScale() const { return scale; }
    uint64_t droppedSteps() const { return dropped; }
    // Threads of parallelEach including the caller (1: serial); between steps
    void setWorkerThreads(int n);
    int workerThreads() const { return pool.workerThreads() + 1; }
    // The time not yet stepped by advance (app state, not saved; a replacement world of a scenario continues with it)
    double accumulatedTime() const { return accumulator; }
    void setAccumulatedTime(double t) { accumulator = t; }

    // FNV-1a 64 over the saved components of the alive entities (index order), the free list, the RNG state, the
    // pending events and the step index (§17.4). Between steps.
    uint64_t hash() const;

    // ---- data (§14, S11) -------------------------------------------------------------------------------------------

    // The entry of a registered data type; throws DataError for an unknown name or type (fatal unless caught: a load
    // and the world API report it)
    template <class T>
    const T& data(const std::string& name) const
    {
        const DataTable<T>* t = dataStore->table<T>();
        if (!t)
            throw DataError(std::string("data: the type ") + T::name + " is not registered (WorldSetup::dataTypes)");
        const T* v = t->find(name);
        if (!v) throw DataError(std::string("data: ") + t->file + " has no entry \"" + name + "\"");
        return *v;
    }
    // Through a DataRef field: resolved once per data store, then without a string lookup
    template <class T>
    const T& data(const DataRef<T>& ref) const
    {
        const uint64_t name = fnv1a64Bytes(ref.entry.data(), ref.entry.size());
        if (ref.resolvedStore.load(std::memory_order_acquire) != dataStore->id() ||
            ref.resolvedName.load(std::memory_order_relaxed) != name)
        {
            const T* entry = &data<T>(ref.entry);
            ref.cached.store(entry, std::memory_order_relaxed);
            ref.resolvedName.store(name, std::memory_order_relaxed);
            ref.resolvedStore.store(dataStore->id(), std::memory_order_release);
            return *entry;
        }
        return *ref.cached.load(std::memory_order_relaxed);
    }
    template <class T>
    bool hasData(const std::string& name) const
    {
        const DataTable<T>* t = dataStore->table<T>();
        return t && t->find(name);
    }
    template <class T>
    std::vector<std::string> dataNames() const
    {
        const DataTable<T>* t = dataStore->table<T>();
        return t ? t->names() : std::vector<std::string>();
    }
    // Reads the data files again: applied only if all are valid and no entry was removed (entries are updated in
    // place: DataRefs and references see the new values; members set in configure() keep theirs until configured
    // again). False with the error and nothing changed otherwise.
    bool reloadData(std::string& error)
    {
        if (dataReadOnly)
        {
            error = "this world shares the data of another world and must not reload it";
            return false;
        }
        return dataStore->reload(error);
    }
    const DataStore& dataTables() const { return *dataStore; }
    // The store itself (worlds of one setup share it: WorldSetup::sharedData)
    std::shared_ptr<DataStore> sharedDataStore() const { return dataStore; }

    // ---- saves (§15, S11) ------------------------------------------------------------------------------------------

    // The saved state as a byte blob (docs/engine_design.md §15.1): the schema table, the slot table, the saved
    // components (Saved::Yes) in dense order, the pending events, the queued actions, the RNG, the Persistent counter,
    // the step index and the time. Deterministic. Between steps, not inside a game hook.
    std::vector<uint8_t> save() const;
    // Replaces the world with a save (§15.2): checks everything first (rejected: the world is unchanged), maps the
    // saved types to the registered ones by name (migrations, removed types), then commits and recreates the game
    // objects (configure, then onLoad; a hook that throws removes its entity, see LoadResult). Between steps, not
    // inside a game hook.
    LoadResult tryLoad(const std::vector<uint8_t>& data);
    // tryLoad; stops the program unless the result is ok()
    void load(const std::vector<uint8_t>& data);
    // The saved components of one entity (or of all, in index order) as JSON, for debugging (§15.1)
    std::string dumpJson(Entity e = Entity{}) const;
    // The differences of the saved state of two worlds, one line each (at most maxLines, then "... and N more");
    // empty if equal. Components are compared by name (diffSaves of a.save() and b.save())
    static std::string diff(const World& a, const World& b, int maxLines = 20);
    // Committed loads so far
    uint64_t loadCount() const { return loads; }
    // Runs fn after every committed load (render caches, the world API); returns an ID for removal
    int addLoadListener(std::function<void(const LoadInfo&)> fn)
    {
        loadListeners.push_back({nextListenerId, std::move(fn)});
        return nextListenerId++;
    }
    void removeLoadListener(int id)
    {
        for (size_t i = 0; i < loadListeners.size(); ++i)
            if (loadListeners[i].first == id) loadListeners.erase(loadListeners.begin() + std::ptrdiff_t(i));
    }
    const MigrationList& migrations() const { return migrationList; }

    // ---- engine-internal -----------------------------------------------------------------------------------------

    Registry& registry() { return reg; }
    const Registry& registry() const { return reg; }
    const ClassList& classList() const { return classes; }
    const SystemList& systemList() const { return systems; }
    const ActionList& actionList() const { return actionNames; }
    // Step timings and counters (§17.1, S10); the app adds frame and GPU numbers from S13
    Profiler& profiler() { return prof; }
    int objectsUpdatedCounter() const { return updatedCounter; }
    const Profiler& profiler() const { return prof; }
    const WorldConfig& worldConfig() const { return config; }
    // Every game object ticks every step regardless of the rings (ScenarioWorld's default, §8.4; S12). Not saved (a
    // test or tool setting); the next tickSystem evaluates every object again when it changes.
    void setTickEveryStep(bool on) { tickAll = on; }
    bool tickEveryStep() const { return tickAll; }
    // The value the last tick evaluation used (gameplay/TickSystem)
    bool tickEveryStepApplied = false;
    // Player cells of the last tick evaluation (space, x, z in TickPolicy::ringSize cells), sorted
    // (gameplay/TickSystem)
    std::vector<std::array<int64_t, 3>> tickPlayerCells;
    StructuralLog& structuralLog() { return log; }
    // Unsorted queryRadius for order-insensitive engine callers (interest sets, ticking): fn(Entity, const Vec3&
    // gridPosition) for each alive entity of the space within radius that has all components of mask, in an
    // unspecified order. gridPosition is the position the decision was made with (the last phase 6), so a caller that
    // needs the distance (hysteresis, nearest player) uses the same state as the query. fn must not change the grid,
    // directly or indirectly (no step(), no rebuildSpatialGrid while the query runs); copy gridPosition to keep it.
    // Nested read-only queries are fine.
    template <class F>
    void forEachWithin(SpaceId space, const Vec3& center, double radius, const ComponentMask& mask, F&& fn) const
    {
        grid.forEachWithin(space, center, radius,
                           [&](const SpatialGrid::Item& it)
                           {
                               // Destroyed since the last update (the slot may be reused: the generation differs)
                               if (!reg.alive(it.entity) || !reg.entityMask(it.entity).containsAll(mask)) return;
                               fn(it.entity, it.position);
                           });
    }
    // The grid of queryRadius (updated by spatialGridSystem in phase 6) and its structural log consumer
    SpatialGrid& spatialGrid() { return grid; }
    const SpatialGrid& spatialGrid() const { return grid; }
    int spatialGridConsumer() const { return gridConsumer; }
    Phase currentPhase() const { return phase; }
    bool inStep() const { return phase != Phase::None; }
    // save() is allowed: between steps and not inside a game hook (save() asserts otherwise); never on a client (S16)
    bool canSave() const { return isHost() && !inStep() && hookDepth == 0 && loadHookPhase == 0; }
    // Entities whose Transform changed in this or the previous step, sorted by index (built in phase 6)
    const std::vector<uint32_t>& changedTransforms() const { return changed; }
    std::vector<uint32_t>& changedTransformsForUpdate() { return changed; }
    // Set when Parent components or entities change: phase 6 rebuilds the depth-sorted child list
    bool hierarchyDirty = true;
    // The entities with a Parent, parents before children (depth, then index); rebuilt by phase 6 when dirty
    std::vector<Entity> hierarchyOrder;
    // Children whose parent was teleported after their last update in a step (phase 8): teleported in the next
    // phase 6, sorted by index
    std::vector<Entity> pendingChildTeleports;
    // ---- type-erased access for the world API (between steps; S10) ----
    // Adds a default component by ID with its log record (not Transform, Behaviour, Persistent or Parent)
    void addDefaultNow(Entity e, ComponentId id);
    // Removes a component by ID with its log record (Parent and Persistent keep their bookkeeping; not Transform)
    void removeByIdNow(Entity e, ComponentId id);
    // Spawns a class by its registration: Transform, Behaviour, then initial(e) adds the initial components, then the
    // tick interval, configure() and onSpawn()
    Entity spawnByClass(const ClassList::ClassInfo& info, SpaceId space, const Vec3& position,
                        const std::function<void(Entity)>& initial);
    // Runs fn after every completed step (also of step(n) and advance); fn returns false to be removed
    int addStepListener(std::function<bool()> fn)
    {
        stepListeners.push_back({nextListenerId, std::move(fn)});
        return nextListenerId++;
    }
    // Removes a step listener (also while the listeners run: it is not called any more)
    void removeStepListener(int id)
    {
        for (size_t i = 0; i < stepListeners.size(); ++i)
            if (stepListeners[i].first == id) stepListeners.erase(stepListeners.begin() + std::ptrdiff_t(i));
        // Only while the listeners run (the running pass must skip it); otherwise nothing is recorded
        if (dispatchingStepListeners) removedStepListeners.push_back(id);
    }
    size_t stepListenerCount() const { return stepListeners.size(); }
    std::vector<int> stepListenerIds() const
    {
        std::vector<int> ids;
        for (const auto& l : stepListeners) ids.push_back(l.first);
        return ids;
    }
    // True while the step listeners run (a listener must not step the world: the dispatch is not re-entrant)
    bool dispatchingListeners() const { return dispatchingStepListeners; }
    // Runs fn when the world is destroyed (observers such as the world API detach); returns an ID for removal
    int addDestroyListener(std::function<void()> fn)
    {
        destroyListeners.push_back({nextListenerId, std::move(fn)});
        return nextListenerId++;
    }
    void removeDestroyListener(int id)
    {
        for (size_t i = 0; i < destroyListeners.size(); ++i)
            if (destroyListeners[i].first == id) destroyListeners.erase(destroyListeners.begin() + std::ptrdiff_t(i));
    }
    // While at least one user asked for it: the alive entities whose Transform changed during the last completed step
    // (any phase), collected during the flag shift at the end of the step (the world API's change journal)
    void retainStepChanges() { trackStepChanges++; }
    void releaseStepChanges() { trackStepChanges = std::max(0, trackStepChanges - 1); }
    const std::vector<Entity>& changedDuringLastStep() const { return stepChanged; }

    // Recounts the children per entity from the Parent pool and marks the hierarchy order dirty (after the registry was
    // filled directly, e.g. by load in S11)
    // Returns false (and changes nothing, in every build: the caller can repair the fill) if a Parent names an entity
    // that is not alive or the Parents form a cycle
    bool rebuildHierarchyState();

    // Tests only: runs in phase 8 right after the replication (an exception aborts the step after the bookkeeping was
    // staged, S16)
    std::function<void()> debugAfterReplication;

   private:
    friend class WorldLoader;
    friend class Replication;
    NetRole role     = NetRole::Host;
    uint8_t selfPeer = 0;
    std::unique_ptr<Replication> repl;
    // The owner of each entity's root by index (ownedBy); rebuilt when dirty
    mutable std::vector<uint8_t> ownerCache;
    mutable bool ownerCacheDirty = true;
    void rebuildOwnerCache() const;
    // Host: an add/remove of Parent or NetOwner that would change the owner of a spawned entity (no transfer after
    // the spawn, §13.3); refused in every build (asserts)
    bool netChangeRefused(Entity e, ComponentId c, bool add, Entity parent);
    void markOwnersDirty() { ownerCacheDirty = true; }
    // Client hook calls (§13.5): a replicated destroy and the addition of a component by ID inside a step
    void addDefaultInternal(Entity e, ComponentId id);
    void removeInternal(Entity e, ComponentId id);
    std::shared_ptr<DataStore> dataStore;
    bool dataReadOnly = false;  // WorldSetup::sharedDataReadOnly
    MigrationList migrationList;
    uint64_t loads = 0;
    std::vector<std::pair<int, std::function<void(const LoadInfo&)>>> loadListeners;
    bool tickAll      = false;
    double timeBase   = 0;
    uint64_t stepBase = 0;
    // The hooks of a load run (1: configure, 2: onLoad of loadHookEntity): structural changes are restricted
    int loadHookPhase   = 0;
    bool loadInProgress = false;  // a load runs (until its listeners ran)
    Entity loadHookEntity;
    // False (and an assert) for a structural change a load hook may not make; refused in every build
    bool checkLoadHook(const char* operation, Entity e, ComponentId id) const;

    // Structural changes are deferred during a step, except while commands are applied
    bool deferring() const { return phase != Phase::None && !flushing; }

    // ---- network (S15)
    struct OutboxEntry
    {
        uint8_t peer;
        Channel channel;
        Reliability reliability;
        Phase phase;  // the phase that sent it (an aborted phase's entries are dropped)
        uint64_t sequence;
        MessageId id;
        std::vector<uint8_t> payload;
    };
    struct ReceivedMessage
    {
        uint8_t peer;
        uint64_t sequence;
        MessageId id;
        std::vector<uint8_t> payload;
    };
    void enqueueMessage(uint8_t peer, const MessageList::Entry& entry, std::vector<uint8_t> payload);
    // Phase 1: polls the transport and runs the handlers in (peer, sequence) order
    // engineOnly (stepNetworkOnly): game messages stay queued for the next step
    void receiveMessages(bool engineOnly = false);
    bool decodePacket(uint8_t peer, const std::vector<uint8_t>& packet, std::vector<ReceivedMessage>& out);
    void rejectInput(uint8_t peer, const std::string& why);
    // A load: discards the outbox (encoded from the replaced state)
    void discardOutboxAfterLoad();
    MessageList messages;
    Transport* net = nullptr;
    std::vector<OutboxEntry> outbox;
    std::vector<ReceivedMessage> inbox;  // the batch of a receive; a leftover of an aborted step is handled first
    std::array<uint64_t, 256> sendSequence{};
    std::vector<uint8_t> peers, joined, left;
    bool peerListsCommitted = true;  // the last receive's step completed (the lists are replaced at the next receive)
    std::array<int, 256> rejectLogs{};
    int sendBatchDepth = 0;
    NetStats netCounts;
    int counterSentBytes = -1, counterReceivedBytes = -1, counterMessagesIn = -1, counterMessagesOut = -1,
        counterRejected = -1;
    void record(std::unique_ptr<Command> c);
    void flushCommands();
    void runStep();
    void checkNotParallel(const char* operation) const;

    void activateNow(Entity e);
    // spawn: the entity (reserved during a step), then at the flush Transform and Behaviour, the initial components,
    // and the start (tick interval, configure, onSpawn)
    Entity beginSpawn(SpaceId space);
    void finishSpawn(Entity e, const ClassList::ClassInfo& info, const Vec3& position);
    void startObject(Entity e);
    // A spawn that failed halfway (an exception in configure, onSpawn or an initial add): the entity is destroyed
    // without its onDestroy, so no half-started object ticks
    void abandonSpawn(Entity e);
    template <class T>
    Entity globalEntity()
    {
        SAIGA_ASSERT(reg.isGlobal(T::id), std::string(T::name) + " is not registered as a global (registerGlobal)");
        return worldState;
    }
    template <class T>
    Entity spaceStateEntityOf(SpaceId space)
    {
        SAIGA_ASSERT(reg.isSpaceState(T::id),
                     std::string(T::name) + " is not registered as space state (registerSpaceState)");
        SAIGA_ASSERT(space < spaceStates.size(), "spaceState: the space " + std::to_string(space) + " is not loaded");
        return spaceStates[space];
    }
    void createStateEntities();
    void deliverActions();
    // add<Parent>: false if the child was destroyed because its parent is gone (during a flush); asserts between steps
    // for a dead parent and always for a cycle
    bool acceptParent(Entity child, Entity parent);
    // On an exception in a step: drops the pending commands and releases the entities they would have activated
    void abortStep();
    // Children per entity index (destroy skips the cascade for entities without children)
    uint32_t childCount(Entity e) const;
    void changeChildCount(Entity parent, int delta);
    // Allocates the parent's count slot (before Parent is added, so the add cannot leave a count missing)
    void reserveChildSlot(Entity parent);
    void destroyNow(Entity e);
    void destroyOne(Entity e);
    std::vector<Entity> collectDoomed(Entity e) const;
    int destroyDepth = 0;              // destroyOne is running (nested destroys are queued)
    std::vector<Entity> destroyQueue;  // destroys requested by hooks during a destroy
    // Game objects of destroyed entities and removed Behaviours, released when no hook runs (a hook may destroy its
    // own entity and keep running)
    std::vector<std::unique_ptr<GameObject>> retired;
    int hookDepth = 0;
    void releaseRetired();
    template <class T>
    void addNow(Entity e, T&& value);
    template <class T>
    void removeNow(Entity e);

    template <class... C>
    static constexpr bool parentReadOnly()
    {
        return (((!std::is_same_v<std::remove_cv_t<C>, Parent> && !std::is_same_v<std::remove_cv_t<C>, Persistent> &&
                  !std::is_same_v<std::remove_cv_t<C>, NetOwner>) ||
                 std::is_const_v<C>) &&
                ...);
    }

    template <class T, class... A>
    static T makeComponent(A&&... args)
    {
        if constexpr (std::is_constructible_v<T, A...>)
            return T(std::forward<A>(args)...);
        else
            return T{std::forward<A>(args)...};
    }

    Registry reg;
    ClassList classes;
    ActionList actionNames;
    Profiler prof;
    // Scope IDs: the step, per phase, per system (in SystemList order), per command flush; counter IDs
    int stepScope = -1;
    std::array<int, size_t(kLastPhase) + 1> phaseScopes{}, flushScopes{};
    std::array<std::vector<int>, size_t(kLastPhase) + 1> systemScopes;
    int entitiesCounter = -1, objectsCounter = -1, commandsCounter = -1, recordsCounter = -1, gridCounter = -1,
        updatedCounter          = -1;
    uint64_t recordsAtStepStart = 0;
    void setupProfiler();
    StructuralLog log;
    SpatialGrid grid;
    int gridConsumer = -1;
    SystemList systems;
    JobPool pool;
    Rng rng;
    std::shared_ptr<const void> contextPtr;
    const std::type_info* contextType = nullptr;
    std::vector<std::unique_ptr<Command>> commands;
    std::vector<Entity> pendingReserved;  // reserved by create() since the last flush
    std::vector<uint32_t> children;       // number of Parent components naming each entity index
    bool stepEventsBegun = false;
    std::vector<uint32_t> changed;
    // Queued actions with their (peer, sequence) key
    struct QueuedAction
    {
        uint8_t peer;
        uint64_t sequence;
        PlayerAction action;
    };
    std::vector<QueuedAction> actions;
    uint64_t actionSequence = 0;
    Entity worldState;
    std::vector<Entity> spaceStates;  // by SpaceId
    std::vector<std::pair<int, std::function<bool()>>> stepListeners;
    std::vector<int> removedStepListeners;  // removed while the listeners run
    bool dispatchingStepListeners = false;
    std::vector<std::pair<int, std::function<void()>>> destroyListeners;
    int nextListenerId   = 0;
    int trackStepChanges = 0;
    std::vector<Entity> stepChanged;
    void runStepListeners();
    uint64_t persistentCounter = 0;
    std::vector<std::pair<Entity, PersistentId>> pendingPersistent;  // assigned in this phase, added at the flush
    std::unordered_map<uint64_t, Entity> persistentIndex;

    WorldConfig config;
    double stepDt;
    double accumulator    = 0;
    double scale          = 1;
    bool paused           = false;
    bool flushing         = false;
    uint64_t steps        = 0;
    uint64_t dropped      = 0;
    double hostStepOffset = 0;  // a client's estimated host step - its step index (HostClock)
    Phase phase           = Phase::None;
};

// ---- template definitions ------------------------------------------------------------------------------------------

template <class T, class... A>
void World::add(Entity e, A&&... args)
{
    static_assert(T::id != ComponentId(EngineComponent::Behaviour), "Behaviour is added by spawn<T>");
    static_assert(T::id != ComponentId(EngineComponent::Persistent), "Persistent is added by makePersistent");
    checkNotParallel("add");
    T value = makeComponent<T>(std::forward<A>(args)...);
    // During a flush an entity (or a parent) reserved later in the phase is not active yet: the add follows its
    // activation
    bool defer = deferring() || (flushing && reg.reserved(e));
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
        defer = defer || (flushing && reg.reserved(value.entity));
    if (defer)
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), std::string("add<") + T::name + ">: the entity is not alive");
        // Whether e has T is checked when the command is applied (earlier commands of the phase may add or remove T:
        // remove + add re-parents, §7.4); the flush asserts with the component and the entity named
        if constexpr (T::id == ComponentId(EngineComponent::Parent))
        {
            // The parent must exist now (it may be destroyed before the flush; then the child follows it); refused in
            // every build
            if (!reg.alive(value.entity) && !reg.reserved(value.entity))
            {
                SAIGA_ASSERT(false, "add<Parent>: the parent is not alive");
                return;
            }
        }
        record(makeCommand([e, v = std::move(value)](World& w) mutable { w.addNow<T>(e, std::move(v)); }));
        return;
    }
    addNow<T>(e, std::move(value));
}

template <class T>
void World::remove(Entity e)
{
    checkNotParallel("remove");
    if (deferring() || (flushing && reg.reserved(e)))
    {
        SAIGA_ASSERT(reg.alive(e) || reg.reserved(e), std::string("remove<") + T::name + ">: the entity is not alive");
        record(makeCommand([e](World& w) { w.removeNow<T>(e); }));
        return;
    }
    removeNow<T>(e);
}

template <class T>
void World::addNow(Entity e, T&& value)
{
    // A command on an entity destroyed earlier in the same flush is dropped
    if (flushing && !reg.alive(e)) return;
    if (loadHookPhase && !checkLoadHook("add", e, T::id)) return;
    // Two deferred adds (hooks see the state before the flush): refused in every build (the pool must not get a
    // second entry)
    if (reg.alive(e) && reg.has<T>(e))
    {
        SAIGA_ASSERT(false, std::string("add<") + T::name + ">: the entity already has " + T::name);
        return;
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        // Ownership is defined on roots (§13.3): an owned root cannot become a child; refused in every build
        if (reg.alive(e) && reg.has<NetOwner>(e))
        {
            SAIGA_ASSERT(false, "add<Parent>: the entity has a NetOwner (ownership is defined on roots, §13.3)");
            return;
        }
        // A parent that is gone: the child follows it (destroyed); a cycle asserts
        if (!acceptParent(e, value.entity)) return;
        if (netChangeRefused(e, T::id, true, value.entity)) return;
    }
    if constexpr (T::id == ComponentId(EngineComponent::NetOwner))
    {
        if (reg.alive(e) && reg.has<Parent>(e))
        {
            SAIGA_ASSERT(false, "NetOwner on a child (ownership is defined on roots, §13.3)");
            return;
        }
        if (netChangeRefused(e, T::id, true, Entity{})) return;
    }
    // Everything that can allocate first (the log record, the parent's count slot); after the add nothing can fail
    auto record = log.prepare(1);
    if constexpr (T::id == ComponentId(EngineComponent::Persistent))
    {
        SAIGA_ASSERT(value.persistentId.valid() && !persistentIndex.count(value.persistentId.value),
                     "Persistent: the ID is invalid or already used");
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        const Entity parent = value.entity;
        reserveChildSlot(parent);
        reg.add<T>(e, std::move(value));
        changeChildCount(parent, +1);
        hierarchyDirty  = true;
        ownerCacheDirty = true;
    }
    else if constexpr (T::id == ComponentId(EngineComponent::Persistent))
    {
        // The index entry first (it allocates), rolled back if the add fails
        const uint64_t id = value.persistentId.value;
        persistentIndex.emplace(id, e);
        try
        {
            reg.add<T>(e, std::move(value));
        }
        catch (...)
        {
            persistentIndex.erase(id);
            throw;
        }
    }
    else
        reg.add<T>(e, std::move(value));
    if constexpr (T::id == ComponentId(EngineComponent::NetOwner)) ownerCacheDirty = true;
    record.append(e, T::id, true);
}

template <class T>
void World::removeNow(Entity e)
{
    if (flushing && !reg.alive(e)) return;
    if (loadHookPhase && !checkLoadHook("remove", e, T::id)) return;
    // Two deferred removes, or none to remove: refused in every build
    if (!reg.alive(e) || !reg.has<T>(e))
    {
        SAIGA_ASSERT(false, std::string("remove<") + T::name + ">: the entity has no " + T::name);
        return;
    }
    if ((reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState)) &&
        (reg.isGlobal(T::id) || reg.isSpaceState(T::id)))
    {
        // Refused in every build: global<T>() and spaceState<T>() rely on them
        SAIGA_ASSERT(false, std::string("remove<") + T::name + ">: world-state components cannot be removed");
        return;
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent) || T::id == ComponentId(EngineComponent::NetOwner))
        if (netChangeRefused(e, T::id, false, Entity{})) return;
    auto record = log.prepare(1);
    if constexpr (T::id == ComponentId(EngineComponent::Behaviour))
    {
        // The object may be running (a hook removing its own Behaviour): released when no hook runs
        retired.reserve(retired.size() + 1);
        if (reg.read<T>(e).object) retired.push_back(std::move(reg.get<T>(e).object));
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent))
    {
        changeChildCount(reg.read<Parent>(e).entity, -1);
        hierarchyDirty = true;
    }
    if constexpr (T::id == ComponentId(EngineComponent::Parent) || T::id == ComponentId(EngineComponent::NetOwner))
        ownerCacheDirty = true;
    if constexpr (T::id == ComponentId(EngineComponent::Persistent))
        persistentIndex.erase(reg.read<T>(e).persistentId.value);
    reg.remove<T>(e);
    record.append(e, T::id, false);
    if constexpr (T::id == ComponentId(EngineComponent::Behaviour)) releaseRetired();
}

}  // namespace Engine
}  // namespace Saiga
