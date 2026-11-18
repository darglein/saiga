/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

// The result of a world API call: ok with a value, or an error message (never an assert: an agent's mistake must not
// stop the game)
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
 * editor use to read and change a running world. Every operation takes a JSON object of arguments and is called between
 * steps (MCP handlers run in McpServer::poll, which the app calls between steps). Entities are written
 * {"index": i, "generation": g} or "i:g"; results always use the object form. Components, classes and events are named
 * as registered; values are their visit fields (VisitJson.h): missing keys keep their values, unknown keys and wrong
 * types are errors and change nothing. Every change is logged under the "api" category.
 *
 * Operations (name: arguments -> result):
 *   describe_world:   -> components, classes, events with their fields
 *   world_status:     -> step, time, paused, time scale, counts, hash
 *   list_entities:    class?, components?: [names], near?: {space?, position, radius}, limit? (100, <= 1000),
 *                     after? (entity index) -> entities [{entity, class?, position, components}], next?
 *   inspect:          entity, components?: [names] -> class, flags, parent, children, components {name: values}
 *   set_component:    entity, component, values -> the new values (Transform: through its setters, a teleport)
 *   add_component:    entity, component, values?      remove_component: entity, component
 *   set_transform:    entity, position?, rotation? [w,x,y,z], scale?, teleport? (true)
 *   spawn:            class?, position, space? (0), components?: {name: values} -> entity
 *   destroy:          entity -> destroyed (the entity and its descendants; onDestroy runs)
 *   get_global:       component, space? (world state if missing)      set_global: component, values, space?
 *   emit_event:       event, values (readable in the next step; the gameplay path, e.g. onDamage)
 *   queue_action:     player, action, aim?
 *   pause, resume, step: n (only while paused), time_scale: scale
 *   changes_since:    cursor (0: from the oldest kept) -> changes [{seq, step, kind, entity, component?}], cursor,
 *                     truncated (kinds: created, destroyed, added, removed, moved)
 *   get_profile:      steps? (window) -> the profiler report      reset_profile
 *   logs:             since?, category? -> lines [{seq, category, message}]
 *   apply:            operations: [{op, ...arguments}] -> one result per operation (applied in order; it stops at
 *                     the first failing operation, the earlier ones stay applied)
 * With allowEdits = false only the reading operations are allowed.
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
    bool editsAllowed() const { return allowEdits; }
    World& world() { return w; }

    // Parses an entity argument ({"index", "generation"} or "i:g")
    static bool parseEntity(const json11::Json& j, Entity& e, std::string& error);
    static json11::Json entityJson(Entity e);

   private:
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
    ApiResult pause(const json11::Json& a);
    ApiResult resume(const json11::Json& a);
    ApiResult step(const json11::Json& a);
    ApiResult timeScale(const json11::Json& a);
    ApiResult changesSince(const json11::Json& a);
    ApiResult getProfile(const json11::Json& a);
    ApiResult resetProfile(const json11::Json& a);
    ApiResult logs(const json11::Json& a);
    ApiResult apply(const json11::Json& a);

    // The change journal of changes_since: filled from an own structural log consumer and the changed list after
    // every step (and before every changes_since), bounded
    struct Change
    {
        uint64_t seq;
        uint64_t step;
        const char* kind;
        Entity entity;
        int component;  // -1: none
    };
    void drainChanges(bool moves);

    World& w;
    bool allowEdits;
    int consumer = -1;
    std::deque<Change> journal;
    uint64_t nextSeq = 1;
    std::shared_ptr<bool> alive;  // the step listener stops when the API is gone
};

}  // namespace Engine
}  // namespace Saiga
