/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/components/CharacterIntent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/net/LoopbackTransport.h"
#include "saiga/engine/world/World.h"
#include "saiga/export.h"

#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Saiga
{
namespace Engine
{
class RecordingRenderSink;

struct ScenarioOptions
{
    uint64_t maxSteps       = 60 * 600;  // 10 min at 60 Hz: more steps fail the scenario
    bool varyThreads        = false;     // Repeat: the second run with 1 worker thread if the setup has more, else 3
    bool strictContinuation = false;     // SaveLoad: the final hash must equal Plain's
    // A wall-clock deadline (run_scenario's tool budget): passing it aborts the run (checked by the observer after each
    // step without touching the world, its RNG or its profile); it never changes the result of a run that finishes.
    // The runner starts no further run of a variant (Repeat's second run and reruns, SaveLoad's run) after it
    std::optional<std::chrono::steady_clock::time_point> deadline;
    // The runner adds the host + client variants (HostClient, HostClientDelay, HostClientAsync; §17.3, S16) for the
    // scenarios that are not local-only. Off for the tech demo games until S16b.
    bool networkVariants = false;
    // Clients of the host + client variants (peers 1..clients)
    int clients = 1;
};

// The network of a host + client run (§17.3): the host and ScenarioOptions::clients client worlds over a
// LoopbackTransport without loss
struct ScenarioNetwork
{
    bool enabled   = false;
    int delaySteps = 0;
    bool async     = false;  // each client starts 3 steps late and runs a seeded 0, 1 or 2 steps per host step
};

// Thrown by the step helpers and the step observation to end a scenario (maxSteps, the runner stopping a rerun);
// caught by the runner. The world API and the world's listener dispatch catch it too (the API returns an error), so
// the abort flag is checked again after every call() and helper step.
struct ScenarioAbort
{
    std::string reason;
};

/**
 * A headless, reproducible world for a scenario (docs/engine_design.md §17.2; S12): typed helpers for spawning,
 * players, input, damage, stepping and checks. Every object ticks every step unless enableDistanceTicking() is called.
 * Profiling is off (no clock calls). The save/load variant replaces the World halfway: call world() each time, never
 * keep the reference across a step helper (entity handles stay valid).
 */
class SAIGA_ENGINE_API ScenarioWorld
{
   public:
    explicit ScenarioWorld(const WorldSetup& setup, const ScenarioOptions& options = {},
                           const ScenarioNetwork& network = {});
    ~ScenarioWorld();
    ScenarioWorld(const ScenarioWorld&)            = delete;
    ScenarioWorld& operator=(const ScenarioWorld&) = delete;

    // The host world (checks run there)
    World& world() { return *w; }
    // The host + client variants (§17.3): the client world of a peer (1..clients) and a host entity's handle there
    // (through its NetId; Entity{} if it has not arrived)
    bool networked() const { return network.enabled; }
    // The peer spawnPlayer uses without an explicit peer (S16b): 1 in the host + client variants (a client's player),
    // else 0. Scenarios that spawn players their own way (the tech demos' spawnDriven) pass it on.
    uint8_t defaultPeer() const { return playerPeer(std::nullopt); }
    World& client(int peer);
    Entity onClient(int peer, Entity hostEntity);
    int clientCount() const { return int(clientWorlds.size()); }
    // The recording render sink of the host world (0; the current world after the save/load swap) or a client (S19b)
    RecordingRenderSink& renderSink(int peer = 0);
    // The world API (created on first use; it follows the swap of the save/load variant)
    WorldApi& api();
    // An API call that also ends the scenario when it ran past maxSteps (prefer it to api().call in scenarios)
    ApiResult call(const std::string& operation, const json11::Json& arguments);

    // A game object (World::spawn)
    template <class T, class... C>
    Entity spawn(const Vec3& position, C&&... initial)
    {
        Entity e = w->spawn<T>(position, std::forward<C>(initial)...);
        involve(e);
        return e;
    }
    // A plain player entity on the host: Transform, Player{peer, characterId 1, 2, ...}, PlayerInput, CharacterIntent
    // (§17.2). Player.peer is peer.value_or(0) in the single-world variants (an explicit peer only sets Player.peer)
    // and peer.value_or(1) in the host + client variants, which also setOwner(player, peer) (peer 0: the host's own
    // player). Does not step: move, input and edit for a client's entity that has not arrived are applied on arrival.
    Entity spawnPlayer(const Vec3& position, std::optional<uint8_t> peer = std::nullopt);
    // A game object player: Player and PlayerInput, no CharacterIntent (the game's controller moves it)
    template <class T, class... C>
    Entity spawnPlayer(const Vec3& position, std::optional<uint8_t> peer, C&&... initial)
    {
        Player p;
        p.peer         = playerPeer(peer);
        p.characterId  = nextCharacterId++;
        const Entity e = spawn<T>(position, p, PlayerInput{}, std::forward<C>(initial)...);
        ownPlayer(e, p.peer);
        return e;
    }
    // PlayerInput.move (shortened to length 1), written on the owner
    void move(Entity player, const Vec3& direction);
    // A one-shot action (world.queueAction on the owner: a PlayerAction message from a client)
    void input(Entity player, uint32_t action, const Vec3& aim = Vec3::Zero());
    // fn(world, entity) on the owner's world with the owner's handle (the way a scenario writes owner-authoritative
    // state); queued until the entity arrived on a client owner
    void edit(Entity entity, std::function<void(World&, Entity)> fn);
    // A DamageEvent (the real path: the target's onDamage in the next step)
    void damage(Entity target, double amount, Entity source = Entity{});
    // The world's ticking policy instead of every object every step
    void enableDistanceTicking();

    void step(int n = 1);
    // A step listener that survives the save/load swap (one added to world() directly is lost with the old world: the
    // swap then fails the scenario)
    void addStepListener(std::function<void()> fn);
    void stepSeconds(double seconds);  // rounded to whole steps
    // Steps until predicate() is true (checked before each step), at most maxSteps; false if not reached (no failure)
    bool stepUntil(const std::function<bool()>& predicate, int maxSteps);

    // Records a failure (with the step) if condition is false; returns condition
    bool check(bool condition, const std::string& message);
    // The distance of the Transform positions (a failed check and infinity if one is not alive)
    double distance(Entity a, Entity b);
    // Entities with T (Transform: every entity except the world-state and space-state entities)
    template <class T>
    size_t count()
    {
        if constexpr (T::id == Transform::id)
        {
            size_t n = w->registry().aliveCount();
            if (w->worldStateEntity().valid()) n--;
            for (SpaceId s = 0; w->spaceStateEntity(s).valid(); ++s) n--;
            return n;
        }
        else
            return w->registry().registered(T::id) ? w->registry().poolById(T::id)->size() : 0;
    }
    // Records T of the entity after every step (shown in the failure report)
    template <class T>
    void trace(Entity e)
    {
        traces.push_back(Trace{e,
                               T::name,
                               [e](World& world) -> json11::Json
                               {
                                   const T* v = world.alive(e) ? world.registry().tryRead<T>(e) : nullptr;
                                   return v ? toJson(*v) : json11::Json();
                               },
                               {}});
        involve(e);
    }
    std::string dumpJson(Entity e) { return w->dumpJson(e); }

    const std::vector<std::string>& failures() const { return failed; }
    // Steps run since the scenario started (by any path: helpers, world().step(), the API)
    uint64_t steps() const { return observedSteps; }

    // ---- runner control (ScenarioRunner) --------------------------------------------------------------------------
    struct Control
    {
        std::vector<uint64_t>* hashes = nullptr;                        // the hash after every step
        uint64_t saveLoadAt    = std::numeric_limits<uint64_t>::max();  // replace the world at this step (helpers)
        bool saveLoadDone      = false;
        uint64_t swappedAtStep = 0;  // the observed steps when the world was replaced
        uint64_t stopAt        = std::numeric_limits<uint64_t>::max();  // save into stopSave and end at this step
        std::vector<uint8_t>* stopSave = nullptr;
    };
    Control control;
    // The save/load round trip of the SaveLoad variant (also called by the runner at the end of short scenarios)
    void saveLoad();
    // The context of a failed run: log lines, dumps of the involved entities, traces
    std::string report() const;
    // Ends the scenario if it was asked to (maxSteps, stopAt); called by the helpers
    void checkAbort() const;
    // The end of a host + client run (§17.3): network-only steps until every client applied a snapshot sent after the
    // scenario ended and the host received the owners' final state, then the comparison of the replicas; failures
    // are recorded. Nothing in the single-world variants. Called by the runner.
    void finishNetwork();

   private:
    uint8_t playerPeer(std::optional<uint8_t> peer) const
    {
        return network.enabled ? peer.value_or(1) : peer.value_or(0);
    }
    void ownPlayer(Entity e, uint8_t peer);
    void stepClients();
    void applyPending();
    void compareReplicas();
    void attach();
    // The step observation is two listeners: the counter first (listeners see steps() including the current step) and
    // the observer last (hash, traces, stop-save and limits see the world after every scenario listener ran)
    void attachCounter();
    void attachObserver();
    // Moves the observer behind listeners added after it
    void moveObserverLast();
    void involve(Entity e);
    void onStep();

    WorldSetup setup;
    ScenarioOptions options;
    ScenarioNetwork network;
    // Before the worlds: the transports and the sinks outlive them
    std::vector<std::unique_ptr<LoopbackTransport>> transports;
    std::vector<std::unique_ptr<RecordingRenderSink>> sinks;
    RecordingRenderSink* hostSink = nullptr;
    std::vector<RecordingRenderSink*> clientSinks;
    RecordingRenderSink* giveSink(World& world);
    std::vector<std::unique_ptr<World>> clientWorlds;
    std::vector<uint64_t> clientSteps;
    uint64_t hostSteps = 0;
    Rng asyncRng;
    struct PendingEdit
    {
        Entity entity;  // the host's handle
        uint8_t peer;
        std::function<void(World&, Entity)> fn;
    };
    std::vector<PendingEdit> pendingEdits;
    std::unique_ptr<World> w;
    std::unique_ptr<WorldApi> apiPtr;
    int stepListener  = -1;         // the observer (last)
    int countListener = -1;         // the counter (first)
    int stepDepth     = 0;          // step helpers running (the swap happens only at depth 0, between listeners)
    std::vector<int> ownListeners;  // IDs in the current world of the listeners of addStepListener
    // One callable per listener, shared by its registrations in the replaced and the replacement world (its captured
    // state survives the swap)
    std::vector<std::shared_ptr<std::function<void()>>> listeners;
    // Replaced worlds live until the scenario ends (a World& kept by mistake, or a listener still running on it,
    // stays valid)
    std::vector<std::unique_ptr<World>> retiredWorlds;
    uint64_t observedSteps   = 0;
    uint64_t nextCharacterId = 1;
    bool tickEverything      = true;
    std::string abortReason;
    uint64_t logStart = 0;
    std::vector<std::string> failed;
    std::vector<Entity> involved;
    struct Trace
    {
        Entity entity;
        const char* component;
        std::function<json11::Json(World&)> read;
        std::deque<std::pair<uint64_t, json11::Json>> values;
    };
    std::vector<Trace> traces;
};

}  // namespace Engine
}  // namespace Saiga
