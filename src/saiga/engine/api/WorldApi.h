/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/save/LoadResult.h"
#include "saiga/engine/world/StructuralLog.h"
#include "saiga/export.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;
class Profiler;

// The result of a world API call: ok with a value, or an error message (never an assert: an agent's mistake must not
// stop the game). A failed batch (apply) also carries the results of the operations that were applied.
struct ApiResult
{
    bool ok = true;
    std::string error;
    json11::Json value;

    static ApiResult success(json11::Json v) { return ApiResult{true, {}, std::move(v)}; }
    static ApiResult failure(std::string message) { return ApiResult{false, std::move(message), {}}; }
};

/**
 * The world as JSON, by names (docs/engine_design.md §16.2, S10): what MCP tools, the scenario helpers (S12) and an
 * editor use to read and change a running world. Every operation takes a JSON object of arguments (unknown argument
 * names are errors) and is called between steps (MCP handlers run in McpServer::poll, which the app calls between
 * steps). Entities are written {"index": i, "generation": g} or "i:g"; results always use the object form. Components,
 * classes and events are named as registered; values are their visit fields (VisitJson.h): missing keys keep their
 * values, unknown keys and wrong types are errors and change nothing. Every change is logged under the "api" category.
 *
 * Operations (name: arguments -> result):
 *   describe_world:   -> components (fields, types, replication, notes), classes, events, operations
 *   world_status:     -> step, time, paused, time scale, counts, hash
 *   list_entities:    class?, components?: [names], near?: {space?, position, radius}, include?: [names],
 *                     limit? (100, <= 1000), after? (entity index) -> entities [{entity, class?, position, components,
 *                     values?, grid_position?, distance?}], count, next?, grid_step? (near selects by the grid of the
 *                     last step)
 *   inspect:          entity, components?: [names] -> class, flags, parent, children, components {name: values}
 *   set_component:    entity, component, values -> values, previous (writable fields: an undo payload; Transform: as
 *                     set_transform; Behaviour: the tick flags, the interval is derived again)
 *   add_component:    entity, component, values?      remove_component: entity, component
 *   set_transform:    entity, position?, rotation? [w,x,y,z], scale?, teleport? (true); not for children
 *   spawn:            class?, position, rotation?, scale?, space? (0), components?: {name: values} (applied after
 *                     onSpawn, merged into components the object added) -> entity, alive
 *   destroy:          entity -> destroyed (the entity and its descendants; onDestroy runs)
 *   get_global:       component, space? (world state if missing)      set_global: component, values, space? -> values,
 *                     previous
 *   emit_event:       event, values -> readable_in_step (the gameplay path, e.g. onDamage)
 *   queue_action:     player, action, aim? -> readable_in_step (a client: forwarded, sent to the host)
 *   pause, resume, step: n (only while paused; stops after 30 s: ran, requested, stopped_early), time_scale: scale
 *   changes_since:    cursor? (opaque; none or 0: from the oldest kept change; "now"), include_moved?, limit? (200) ->
 *                     changes [{seq, step, kind, entity, component?}] (kinds: created, destroyed (its components with
 *                     it), added, removed; step: the steps completed when the change was read), moved? [entities],
 *                     moved_more?, moves_tracked_from_step?, cursor, more, truncated, step
 *   get_profile:      steps? (window), scopes? [path prefixes], frames? (window of the app's profiles) -> the step
 *                     profile, plus the app's profiles under their keys (frame, gpu)      reset_profile
 *   logs:             since?, category? -> lines [{seq, category, message}], first, last, truncated
 *   save_snapshot:    name, overwrite? -> name, bytes, step, hash (World::save kept in the API's memory: at most 16
 *                     snapshots and 256 MB; reading, the world does not change)
 *   load_snapshot:    name, paused? -> step, hash, entities, paused, world_generation, hook_errors (World::tryLoad)
 *   list_snapshots:   -> snapshots [{name, bytes, step, hash}]      delete_snapshot: name
 *   diff_snapshot:    name, against? (another snapshot; default the live world), entity?, components?, limit? (20),
 *                     after? -> summary, differences [{kind, entity?, component?, field?, a, b, class?, position?}],
 *                     total, next?
 *   export_snapshot, import_snapshot: name, file (a plain file name in the snapshot directory), overwrite?
 *   get_data:         type, name? -> entry values, or the entry names
 *   reload_data:      reconfigure? (true) -> types, entries, hook_errors (re-reads the data files; with reconfigure
 *                     the world is saved and loaded so every game object runs configure() again)
 *   apply:            operations: [{op, ...arguments}] -> one result per operation, applied in order; it stops at the
 *                     first failing operation (the earlier ones stay applied). Allowed read-only if every
 *                     operation reads.
 * With allowEdits = false only the reading operations are allowed. The API detaches when the world is destroyed
 * (every call then fails). A committed load (load_snapshot, World::tryLoad from anywhere) appears in changes_since as
 * one entry {kind: "world_loaded", step, from_step, world_generation}; a cursor from before it stays valid and its
 * answer has reloaded: true (re-list the entities: handles from after the snapshot may now name other entities).
 */
class SAIGA_ENGINE_API WorldApi
{
   public:
    explicit WorldApi(World& world, bool allowEdits = true);
    ~WorldApi();
    WorldApi(const WorldApi&)            = delete;
    WorldApi& operator=(const WorldApi&) = delete;

    // Runs one operation by name
    ApiResult call(const std::string& operation, const json11::Json& arguments);
    // The names of the operations (the reading ones only without edits)
    std::vector<std::string> operations() const;
    static bool isEdit(const std::string& operation);
    // The argument names of an operation (the tool schemas list exactly these)
    static std::vector<std::string> arguments(const std::string& operation);
    bool editsAllowed() const { return allowEdits; }
    // The wall-clock budget of one request in seconds (step and apply stop when it ends; default 30). The MCP tools
    // set the request's real deadline for every call instead (setDeadline).
    void setTimeBudget(double seconds)
    {
        budget      = seconds;
        hasDeadline = false;
    }
    void setDeadline(std::chrono::steady_clock::time_point d)
    {
        deadline    = d;
        hasDeadline = true;
    }
    // nullptr after the world was destroyed
    World* world() { return w; }
    // The API's step listener in its world (it fills the journal)
    int stepListenerId() const { return stepListener; }
    // Moves the API to another world (S12's save/load variant): cursors, snapshots and the change journal stay; the
    // journal gets a world_loaded entry
    void rebind(World& world);
    // A profile of the app reported by get_profile under key (S13: "frame", "gpu"); it must outlive the API
    void addProfile(const std::string& key, Profiler* profiler) { profiles.push_back({key, profiler}); }
    const std::vector<std::pair<std::string, Profiler*>>& extraProfiles() const { return profiles; }
    // The engine app's co-op mode (--coop, S16b): the API is bound to the client world; pause, resume, time_scale and
    // step act on both worlds (step through stepAll: host one step, then the client one step), reload_data reloads
    // the shared data and reconfigures through a host save and load (the client resyncs), the snapshot operations are
    // refused (a client cannot save) and world_status and get_profile add the host's numbers under "host". The host
    // world must outlive the API's use of it.
    struct CoopHooks
    {
        World* host = nullptr;
        std::function<void(int steps)> stepAll;
    };
    void setCoop(CoopHooks hooks) { coopHooks = std::move(hooks); }
    const CoopHooks& coop() const { return coopHooks; }
    bool inCoop() const { return coopHooks.host != nullptr; }
    // The host block of world_status in co-op (step, time, paused, time scale, counts, dropped steps, hash)
    static json11::Json hostStatus(const World& host);
    // The directory of export_snapshot / import_snapshot (empty: refused)
    void setSnapshotDirectory(const std::string& directory) { snapshotDirectory = directory; }
    // Stores given save bytes as the snapshot name (run_scenario, S14), with save_snapshot's limits (count, bytes) and
    // a new revision; result as save_snapshot's
    ApiResult putSnapshot(const std::string& name, std::vector<uint8_t> save, uint64_t step, uint64_t hash,
                          bool overwrite);
    bool hasSnapshot(const std::string& name) const { return snapshots.count(name) != 0; }
    size_t snapshotCount() const { return snapshots.size(); }
    static constexpr size_t kMaxSnapshotCount = 16;

    // Parses an entity argument ({"index", "generation"} or "i:g")
    static bool parseEntity(const json11::Json& j, Entity& e, std::string& error);
    static json11::Json entityJson(Entity e);

   private:
    ApiResult run(const std::string& operation, const json11::Json& a);
    ApiResult describeWorld(const json11::Json& a);
    ApiResult worldStatus(const json11::Json& a);
    ApiResult listEntities(const json11::Json& a);
    ApiResult inspect(const json11::Json& a);
    ApiResult setComponent(const json11::Json& a);
    ApiResult addComponent(const json11::Json& a);
    ApiResult removeComponent(const json11::Json& a);
    ApiResult setTransform(const json11::Json& a);
    ApiResult spawn(const json11::Json& a);
    ApiResult destroy(const json11::Json& a);
    ApiResult getGlobal(const json11::Json& a);
    ApiResult setGlobal(const json11::Json& a);
    ApiResult emitEvent(const json11::Json& a);
    ApiResult queueAction(const json11::Json& a);
    ApiResult step(const json11::Json& a);
    ApiResult timeScale(const json11::Json& a);
    ApiResult changesSince(const json11::Json& a);
    ApiResult getProfile(const json11::Json& a);
    ApiResult logs(const json11::Json& a);
    ApiResult apply(const json11::Json& a);
    ApiResult saveSnapshot(const json11::Json& a);
    ApiResult loadSnapshot(const json11::Json& a);
    ApiResult listSnapshots(const json11::Json& a);
    ApiResult deleteSnapshot(const json11::Json& a);
    ApiResult diffSnapshot(const json11::Json& a);
    ApiResult exportSnapshot(const json11::Json& a);
    ApiResult importSnapshot(const json11::Json& a);
    ApiResult getData(const json11::Json& a);
    ApiResult reloadData(const json11::Json& a);
    void attach();
    void detach();
    void onLoaded(const LoadInfo& info);
    static json11::Json hookErrorsJson(const LoadResult& r);
    static json11::Json listenerErrorsJson(const LoadResult& r);
    static json11::Json removedJson(const LoadResult& r);
    void markLoaded(uint64_t step, uint64_t fromStep);
    void journalRecord(const StructuralLog::Record& r, uint64_t stepIndex);

    // The change journal of changes_since: structural changes from an own structural log consumer (drained after every
    // step and before every changes_since; the components of a destroyed entity are folded into "destroyed"),
    // bounded; moves are tracked per entity index (the step of its last move) once a caller asked for them
    struct Change
    {
        uint64_t seq;
        uint64_t step;
        const char* kind;
        Entity entity;
        int component;  // -1: none
        // world_loaded: the step before the load and the world's load count
        uint64_t fromStep = 0, generation = 0;
    };
    void drainChanges();
    void trackMoves();

    World* w;
    bool allowEdits;
    int consumer         = -1;
    int destroyListener  = -1;
    int stepListener     = -1;
    int loadListener     = -1;
    uint64_t lastLoadSeq = 0;  // the journal entry of the last load (cursors before it get reloaded: true)
    // Loads and world swaps this API has seen (world_status and the world_loaded entries report it; it only grows)
    uint64_t generation = 0;
    struct Snapshot
    {
        std::vector<uint8_t> bytes;
        uint64_t step = 0, hash = 0;
        bool hasHash      = true;  // false for imported snapshots
        uint64_t revision = 0;     // unique per stored snapshot (a replacement under the same name has another)
    };
    uint64_t nextSnapshotRevision = 1;
    std::map<std::string, Snapshot> snapshots;
    std::string snapshotDirectory;
    std::vector<std::pair<std::string, Profiler*>> profiles;
    CoopHooks coopHooks;
    std::deque<Change> journal;
    uint64_t nextSeq = 1;
    uint32_t epoch   = 0;
    // apply: one wall-clock deadline for the whole batch (step inside it does not start a new budget)
    bool inBatch     = false;
    double budget    = 30;
    bool hasDeadline = false;
    std::chrono::steady_clock::time_point deadline;
    std::chrono::steady_clock::time_point budgetDeadline() const;
    std::chrono::steady_clock::time_point batchDeadline;
    Entity lastDestroyed;  // folds the removed records that follow a destroyed record
    bool movesTracked   = false;
    uint64_t movesSince = 0;  // the step from which moves are tracked
    // By entity index: the handle and the steps of its last two moves since tracking started (0: none). Two steps: an
    // entity that moves again while a window is paged stays in that window (its earlier move is in it)
    struct MoveRecord
    {
        Entity entity;
        uint64_t previous = 0, step = 0;
        bool in(uint64_t from, uint64_t to) const
        {
            return (step > from && step <= to) || (previous > from && previous <= to);
        }
    };
    std::vector<MoveRecord> lastMoved;
    // The members of move windows that are being paged, fixed when the window opened (every entity that moved in the
    // window is delivered, also if it moves again meanwhile); the newest 16 windows are kept
    struct WindowSnapshot
    {
        uint64_t from = 0, to = 0;
        std::vector<Entity> members;  // index order
        bool truncated = false;       // rebuilt after eviction: may miss entities (every cursor is told)
    };
    std::deque<WindowSnapshot> windows;
    std::shared_ptr<bool> alive;  // the world's listeners stop when the API is gone
};

}  // namespace Engine
}  // namespace Saiga
