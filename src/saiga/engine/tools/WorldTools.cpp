/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// registerWorldTools: the engine's MCP tools over the world API, and measure_performance.

#include "saiga/engine/tools/WorldTools.h"

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/world/World.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace Saiga
{
namespace Engine
{
namespace
{
using Json       = json11::Json;
using ToolResult = McpServer::ToolResult;

Json entityProperty(const std::string& description)
{
    // A type list instead of oneOf (some MCP clients reject oneOf in input schemas)
    return Json::object{{"type", Json::array{"object", "string"}},
                        {"description", description + " ({\"index\": i, \"generation\": g} or \"i:g\")"}};
}
Json objectProperty(const std::string& description)
{
    return Json::object{{"type", "object"}, {"description", description}};
}
Json arrayProperty(const std::string& description, const Json& items)
{
    return Json::object{{"type", "array"}, {"description", description}, {"items", items}};
}
Json quatProperty(const std::string& description)
{
    return Json::object{{"type", "array"},
                        {"description", description},
                        {"items", Json::object{{"type", "number"}}},
                        {"minItems", 4},
                        {"maxItems", 4}};
}

struct ToolSpec
{
    const char* name;
    const char* description;
    Json schema;
};

std::vector<ToolSpec> toolSpecs()
{
    using S         = McpServer;
    const Json comp = S::StringProperty("Component name (describe_world lists them)");
    return {
        {"describe_world",
         "Reads the registrations: every component (fields with types, replication, global/space state), game object "
         "class and event. Start here to learn the names and fields the other tools take. Changes nothing.",
         S::ObjectSchema({})},
        {"world_status",
         "Reads step index, simulation time, paused, time scale, entity and game object counts and the "
         "world hash. Changes nothing.",
         S::ObjectSchema({})},
        {"list_entities",
         "Lists entities in index order with class, position and component names. Filters: class, components (all "
         "required), near {space, position, radius} (uses the positions of the last step: entities spawned or moved "
         "since are found after the next step; results then have distance). include: component names whose values "
         "are returned per entity (values). Paged: limit (default 100, at most 1000), after (the index of the last "
         "entity of the previous page; the result has next when more exist). Changes nothing.",
         S::ObjectSchema(
             {{"class", S::StringProperty("Class name")},
              {"components", arrayProperty("Component names", Json::object{{"type", "string"}})},
              {"near", objectProperty("{space?: 0, position: [x,y,z], radius}")},
              {"include", arrayProperty("Return these components' values", Json::object{{"type", "string"}})},
              {"limit", S::IntegerProperty("Maximum entities (1..1000, default 100)")},
              {"after", S::IntegerProperty("Entity index to continue after")}})},
        {"inspect",
         "Reads one entity: class, flags (simulated, world/space state), parent, children and all components (or the "
         "named ones) as JSON. Changes nothing.",
         S::ObjectSchema({{"entity", entityProperty("The entity")},
                          {"components", arrayProperty("Only these components", Json::object{{"type", "string"}})}},
                         {"entity"})},
        {"set_component",
         "Changes fields of a component the entity has (only the given fields; others keep their values). Bypasses "
         "game rules (e.g. Health.hp directly; emit_event with DamageEvent goes through onDamage instead). Transform: "
         "position, rotation, scale (a teleport). Returns the new values and the previous ones (for an undo).",
         S::ObjectSchema({{"entity", entityProperty("The entity")},
                          {"component", comp},
                          {"values", objectProperty("Fields to change")}},
                         {"entity", "component", "values"})},
        {"add_component",
         "Adds a component (default values, then the given fields). Not Transform, Behaviour (spawn a class) or "
         "Persistent. Parent: {entity} must be alive and must not form a cycle. Returns the values.",
         S::ObjectSchema({{"entity", entityProperty("The entity")},
                          {"component", comp},
                          {"values", objectProperty("Initial fields")}},
                         {"entity", "component"})},
        {"remove_component",
         "Removes a component (not Transform; removing Behaviour drops the game object without onDestroy).",
         S::ObjectSchema({{"entity", entityProperty("The entity")}, {"component", comp}}, {"entity", "component"})},
        {"set_transform",
         "Moves an entity: position [x,y,z], rotation [w,x,y,z] (normalized), scale [x,y,z]; missing ones keep their "
         "values. teleport (default true): no interpolation, children follow at once.",
         S::ObjectSchema({{"entity", entityProperty("The entity")},
                          {"position", S::Vec3Property("Position in meters")},
                          {"rotation", quatProperty("Rotation quaternion [w,x,y,z]")},
                          {"scale", S::Vec3Property("Scale")},
                          {"teleport", S::BooleanProperty("Teleport (default true)")}},
                         {"entity"})},
        {"spawn",
         "Creates an entity at position with rotation [w,x,y,z] and scale: with class a game object (configure and "
         "onSpawn run), else a plain entity. components {name: fields} are checked before anything is created and "
         "applied after onSpawn (merged into components the object added itself). Returns the entity.",
         S::ObjectSchema({{"class", S::StringProperty("Game object class (optional)")},
                          {"position", S::Vec3Property("Position in meters")},
                          {"rotation", quatProperty("Rotation quaternion [w,x,y,z] (identity)")},
                          {"scale", S::Vec3Property("Scale ([1,1,1])")},
                          {"space", S::IntegerProperty("Space (0)")},
                          {"components", objectProperty("{component name: fields}")}},
                         {"position"})},
        {"destroy",
         "Destroys an entity and its descendants (their onDestroy runs). World-state entities cannot be destroyed. "
         "Returns the destroyed entities.",
         S::ObjectSchema({{"entity", entityProperty("The entity")}}, {"entity"})},
        {"get_global", "Reads a world-state component (or a space-state component with space). Changes nothing.",
         S::ObjectSchema({{"component", comp}, {"space", S::IntegerProperty("Space for space state")}}, {"component"})},
        {"set_global", "Changes fields of a world-state component (or a space-state component with space).",
         S::ObjectSchema({{"component", comp},
                          {"values", objectProperty("Fields to change")},
                          {"space", S::IntegerProperty("Space for space state")}},
                         {"component", "values"})},
        {"emit_event",
         "Emits an event (fields as values); it is readable in the next step, so the game reacts through its rules "
         "(e.g. DamageEvent {target, source, amount} reaches the target's onDamage).",
         S::ObjectSchema({{"event", S::StringProperty("Event name")}, {"values", objectProperty("Event fields")}},
                         {"event"})},
        {"queue_action",
         "Queues a one-shot player action (game-defined code, optional aim [x,y,z]); delivered as ActionEvent in the "
         "next step, in (peer, sequence) order, exactly once.",
         S::ObjectSchema({{"player", entityProperty("The player entity")},
                          {"action", S::IntegerProperty("Action code")},
                          {"aim", S::Vec3Property("Aim point")}},
                         {"player", "action"})},
        {"pause", "Pauses the simulation (frames continue).", S::ObjectSchema({})},
        {"resume", "Resumes the simulation.", S::ObjectSchema({})},
        {"step", "Runs exactly n steps (default 1, at most 100000); only while paused. Returns the step index.",
         S::ObjectSchema({{"n", S::IntegerProperty("Number of steps")}})},
        {"time_scale", "Sets the simulation speed (0 stops it, 1 real time).",
         S::ObjectSchema({{"scale", S::NumberProperty("Time scale >= 0")}}, {"scale"})},
        {"changes_since",
         "Reads what changed since a cursor: created, destroyed (its components with it), component added/removed, "
         "with sequence number and step; with include_moved also the entities moved since (once each; tracking starts "
         "with the first such call). Pass the returned cursor next time (0: everything kept, \"now\": only what "
         "happens from now on). more / moved_more: call again; truncated: changes were dropped (more than 100000 since "
         "the cursor). Changes nothing.",
         S::ObjectSchema(
             {{"cursor", Json::object{{"type", Json::array{"string", "integer"}},
                                      {"description", "Cursor of the previous call, 0 or \"now\""}}},
              {"include_moved", S::BooleanProperty("Also report moved entities")},
              {"limit", S::IntegerProperty("Maximum changes and moved entities per call (1..1000, 200)")}})},
        {"get_profile",
         "Reads the step profile: per scope (step, phases, systems, command flushes) avg/min/p50/p95/max in ms and "
         "the share of the parent, plus counters (entities, game objects, objects updated, commands, structural "
         "records, grid items), over the last steps (default: the whole history of 600). Changes nothing.",
         S::ObjectSchema({{"steps", S::IntegerProperty("Window in steps (default all kept)")},
                          {"scopes", arrayProperty("Only scopes with these path prefixes (e.g. step/phase:behaviour)",
                                                   Json::object{{"type", "string"}})}})},
        {"reset_profile", "Drops the profile history.", S::ObjectSchema({})},
        {"logs",
         "Reads the engine log lines after a sequence number (at most the newest 500), optionally of one "
         "category (\"api\" lists the changes made through these tools). Changes nothing.",
         S::ObjectSchema({{"since", S::IntegerProperty("Sequence number (0: all kept)")},
                          {"category", S::StringProperty("Category filter")}})},
        {"apply",
         "Runs a batch: operations [{op: tool name, ...arguments}] in order, one result each. Stops at the first "
         "failing operation (the earlier ones stay applied; the error names it).",
         S::ObjectSchema({{"operations", arrayProperty("Operations", Json::object{{"type", "object"}})}},
                         {"operations"})},
    };
}

// Compact JSON with exact numbers: integers in full, others in the shortest form that reads back to the same double
// (McpServer::FormatJson shortens to 6 digits, too few for values an agent reads and writes back)
void formatExact(const Json& j, std::string& out)
{
    switch (j.type())
    {
        case Json::NUMBER:
        {
            const double v = j.number_value();
            if (!std::isfinite(v))
            {
                out += "null";
                return;
            }
            char buf[40];
            if (v == std::floor(v) && std::abs(v) < 9007199254740992.0)
            {
                std::snprintf(buf, sizeof(buf), "%.0f", v);
                out += buf;
                return;
            }
            for (int precision = 15; precision <= 17; ++precision)
            {
                std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
                if (std::strtod(buf, nullptr) == v) break;
            }
            out += buf;
            return;
        }
        case Json::ARRAY:
        {
            out += "[";
            bool first = true;
            for (const Json& e : j.array_items())
            {
                if (!first) out += ",";
                first = false;
                formatExact(e, out);
            }
            out += "]";
            return;
        }
        case Json::OBJECT:
        {
            out += "{";
            bool first = true;
            for (const auto& kv : j.object_items())
            {
                if (!first) out += ",";
                first = false;
                out += Json(kv.first).dump();
                out += ":";
                formatExact(kv.second, out);
            }
            out += "}";
            return;
        }
        default:
            out += j.dump();
    }
}

std::string exactJson(const Json& j)
{
    std::string out;
    formatExact(j, out);
    return out;
}

ToolResult toResult(const ApiResult& r)
{
    if (r.ok) return ToolResult::Text(exactJson(r.value));
    ToolResult t = ToolResult::Error(r.error);
    if (!r.value.is_null()) t.addText(exactJson(r.value));
    return t;
}
}  // namespace

std::shared_ptr<WorldApi> registerWorldTools(McpServer& server, World& world, const WorldToolsOptions& options)
{
    auto api                     = std::make_shared<WorldApi>(world, options.allowEdits);
    std::weak_ptr<WorldApi> weak = api;
    for (ToolSpec& spec : toolSpecs())
    {
        // apply stays available read-only (the API allows batches of reads)
        if (WorldApi::isEdit(spec.name) && !options.allowEdits && std::string(spec.name) != "apply") continue;
        const std::string op = spec.name;
        // Unknown arguments are errors (the API checks them too)
        Json::object schema            = spec.schema.object_items();
        schema["additionalProperties"] = false;
        server.addTool(spec.name, spec.description, schema,
                       [weak, op](const Json& args, McpServer::Respond respond)
                       {
                           auto a = weak.lock();
                           if (!a) return respond(ToolResult::Error("the world is not available"));
                           respond(toResult(a->call(op, args)));
                       });
    }

    // Runs while paused (exactly n steps, within a wall-clock budget), or waits for n steps while running (answered
    // before the tool call times out, with the steps measured so far)
    const double timeout = server.toolTimeoutSeconds;
    server.addTool(
        "measure_performance",
        "Measures n steps (default 120, at most 600: the profile history) and returns the step profile of exactly "
        "those "
        "steps (see get_profile). Paused: runs the steps now (a pure simulation benchmark; edits must be allowed). "
        "Running: waits until the app has run them (refused at time scale 0); it answers early with fewer steps if "
        "they take too long. Changes nothing but time while paused.",
        [&]()
        {
            Json::object schema =
                McpServer::ObjectSchema({{"steps", McpServer::IntegerProperty("Number of steps (1..600)")}})
                    .object_items();
            schema["additionalProperties"] = false;
            return Json(schema);
        }(),
        [weak, timeout](const Json& args, McpServer::Respond respond)
        {
            auto a = weak.lock();
            if (!a || !a->world()) return respond(ToolResult::Error("the world is not available"));
            for (auto& kv : args.object_items())
                if (kv.first != "steps") return respond(ToolResult::Error("unknown argument \"" + kv.first + "\""));
            World& w       = *a->world();
            const double n = args["steps"].is_null() ? 120 : args["steps"].number_value();
            const int max  = w.profiler().historySize();
            if ((!args["steps"].is_null() && !args["steps"].is_number()) || !(n >= 1 && n <= max) || n != std::floor(n))
                return respond(ToolResult::Error("steps: an integer 1.." + std::to_string(max)));
            if (w.inStep()) return respond(ToolResult::Error("measure_performance: only between steps"));
            const auto t0 = std::chrono::steady_clock::now();
            auto elapsed  = [t0]()
            { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
            if (w.isPaused())
            {
                if (!a->editsAllowed())
                    return respond(ToolResult::Error("measure_performance: running steps while paused is an edit"));
                int ran = 0;
                while (ran < int(n) && elapsed() < 0.5 * timeout)
                {
                    w.step();
                    ran++;
                }
                return respond(ToolResult::Text(exactJson(w.profiler().report(ran))));
            }
            if (w.timeScale() == 0)
                return respond(ToolResult::Error("measure_performance: the time scale is 0 (no steps run)"));
            const uint64_t start = w.profiler().sampleCount();
            // Answered exactly once: after n steps, at the deadline, or when the world is destroyed meanwhile
            auto answered     = std::make_shared<bool>(false);
            const int destroy = w.addDestroyListener(
                [respond, answered]()
                {
                    if (*answered) return;
                    *answered = true;
                    respond(ToolResult::Error("the world was destroyed during the measurement"));
                });
            w.addStepListener(
                [weak, respond, start, n, elapsed, timeout, answered, destroy]()
                {
                    if (*answered) return false;
                    auto api = weak.lock();
                    if (!api || !api->world())
                    {
                        *answered = true;
                        respond(ToolResult::Error("the world is not available"));
                        return false;
                    }
                    World& world        = *api->world();
                    const uint64_t done = world.profiler().sampleCount() - start;
                    if (done < uint64_t(n) && elapsed() < 0.8 * timeout) return true;
                    *answered = true;
                    world.removeDestroyListener(destroy);
                    respond(ToolResult::Text(exactJson(world.profiler().report(int(done)))));
                    return false;
                });
        });
    return api;
}

}  // namespace Engine
}  // namespace Saiga
