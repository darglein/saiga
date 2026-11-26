/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// registerWorldTools: the engine's MCP tools over the world API, and measure_performance.

#include "saiga/engine/tools/WorldTools.h"

#include "saiga/core/util/mcp/McpServer.h"
#include "saiga/engine/api/WorldApi.h"
#include "saiga/engine/core/Profiler.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <charconv>
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

// A point [x, y, z] or [x, z] (edit_terrain's center)
Json pointProperty(const std::string& description)
{
    return Json::object{{"type", "array"},
                        {"description", description},
                        {"items", Json::object{{"type", "number"}}},
                        {"minItems", 2},
                        {"maxItems", 3}};
}

struct ToolSpec
{
    const char* name;
    const char* description;
    Json schema;
};

// A chunk [x, z] (two integers)
json11::Json chunkProperty(const std::string& description)
{
    return json11::Json::object{{"type", "array"},
                                {"items", json11::Json::object{{"type", "integer"}}},
                                {"minItems", 2},
                                {"maxItems", 2},
                                {"description", description}};
}

// A position [x, z] or [x, y, z] (numbers)
json11::Json positionXzProperty(const std::string& description)
{
    return json11::Json::object{{"type", "array"},
                                {"items", json11::Json::object{{"type", "number"}}},
                                {"minItems", 2},
                                {"maxItems", 3},
                                {"description", description}};
}

std::vector<ToolSpec> toolSpecs()
{
    using S         = McpServer;
    const Json comp = S::StringProperty("Component name (describe_world lists them)");
    return {
        {"describe_world",
         "Reads the registrations: every component (fields with types, replication, global/space state), game object "
         "class and event, and the spaces (id, name, kind, size_chunks, has_terrain, sea_level with terrain, "
         "site_origin of an enclosed space), and the content generators (name, version, the pinned plan and current "
         "placement versions, the slots with their flags, integer_only). Start here to learn the names and fields "
         "the other tools take. Changes nothing.",
         S::ObjectSchema({})},
        {"world_status",
         "Reads step index, simulation time, paused, time scale, entity and game object counts and the "
         "world hash, the content generation {seed (a string), mismatch, checked: a client's check of the host's "
         "generation state}, per space (spaces) the entity count and the players in it, streaming {enabled, resident "
         "and "
         "pending chunks, stalls, islands, simulated entities}; in an app with --coop (the "
         "tools see the client world) also host: the hidden host world's step, time, paused, time scale, counts, "
         "dropped steps and hash. Changes nothing.",
         S::ObjectSchema({})},
        {"list_entities",
         "Lists entities in index order with class, position and component names. Filters: class, components (all "
         "required), space (the entities of one space, without near), near {space, position, radius} (uses the "
         "positions of the last step: entities spawned or moved since are found after the next step; results then "
         "have distance). include: component names whose values "
         "are returned per entity (values). Paged: limit (default 100, at most 1000), after (the index of the last "
         "entity of the previous page; the result has next when more exist). Changes nothing.",
         S::ObjectSchema(
             {{"class", S::StringProperty("Class name")},
              {"components", arrayProperty("Component names", Json::object{{"type", "string"}})},
              {"near", objectProperty("{space?: 0, position: [x,y,z], radius}")},
              {"space", S::IntegerProperty("Only the entities of this space (not with near)")},
              {"include", arrayProperty("Return these components' values", Json::object{{"type", "string"}})},
              {"limit", S::IntegerProperty("Maximum entities (1..1000, default 100)")},
              {"after", S::IntegerProperty("Entity index to continue after")}})},
        {"inspect",
         "Reads one entity: class, flags (simulated, world/space state), held (its root waits for a streaming arrival: "
         "frozen and hidden), parent, children and all components (or the named ones) as JSON. Changes nothing.",
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
        {"move_to_space",
         "Moves a root and its descendants to another space (or within its space) at position [x,y,z] with rotation "
         "[w,x,y,z] (default: the current one): a teleport; for a client-owned root on the host the owner gets the "
         "move as a command. Not children (move their root) or state entities. Returns the entity and previous "
         "{space, position, rotation} (sent back as a move_to_space it restores the entity).",
         S::ObjectSchema({{"entity", entityProperty("The root entity")},
                          {"space", S::IntegerProperty("Target space (describe_world lists them)")},
                          {"position", S::Vec3Property("Position in meters in the target space")},
                          {"rotation", quatProperty("Rotation quaternion [w,x,y,z] (default: the current one)")}},
                         {"entity", "space", "position"})},
        {"request_arrival",
         "Requests an arrival of a root at position [x,y,z] in a space (default: its current one) with rotation "
         "[w,x,y,z] (default: the current one); host only, between steps. In a streaming world a player root is held "
         "(frozen, hidden, its pose unchanged) and the target's chunks load first; it moves in phase 1 of the step "
         "after "
         "its island (the chunks within simulationRadius of the target) is resident. Other roots and worlds without "
         "streaming arrive in the next step. Refused (an error with the reason): an arrival already pending, the hard "
         "island limit, a position outside the space or not finite, an unknown space, a child or state entity. Returns "
         "accepted, the target and the island progress (0..1). Changes: adds PendingTransition.",
         S::ObjectSchema({{"entity", entityProperty("The root entity")},
                          {"space", S::IntegerProperty("Target space (default: the entity's)")},
                          {"position", S::Vec3Property("Position in meters in the target space")},
                          {"rotation", quatProperty("Rotation quaternion [w,x,y,z] (default: the current one)")}},
                         {"entity", "position"})},
        {"edit_terrain",
         "Terraforms the height-map terrain of an open-world space (host only, between steps; the change is "
         "replicated to the clients and saved): op \"add\" / \"subtract\" (strength: metres at the center, 0..64) "
         "or \"flatten\" / \"smooth\" (strength: blend at the center, 0..1); center [x, y, z] in metres (y: the "
         "flatten target, 0 when omitted; [x, z] is accepted), radius 0..128 m, quartic falloff. Changes the terrain "
         "heights and tombstones the generated instances of tombstone_on_edit slots whose surface it changes. "
         "Returns changed_samples, per changed chunk {chunk: [x, z], sequence, box: [x0, z0, w, h] in chunk-local "
         "samples}, the changed box [x0, z0, x1, z1] in metres and tombstoned [{chunk: [x, z], slot, counter}]. "
         "Refused on clients, inside a step, "
         "for an unknown or enclosed space and for a brush out of its bounds (the error names the field).",
         S::ObjectSchema(
             {{"space", S::IntegerProperty("Space (default 0; describe_world lists them)")},
              {"op", S::StringProperty("add, subtract, flatten or smooth")},
              {"center", pointProperty("Brush center [x, y, z] or [x, z] in metres (y: the flatten target)")},
              {"radius", S::NumberProperty("Radius in metres (0..128]")},
              {"strength", S::NumberProperty("Metres (add/subtract) or blend 0..1 (flatten/smooth)")}},
             {"op", "center", "radius", "strength"})},
        {"terrain_height",
         "Reads the exact terrain surface height at (x, z) of a space (the integer surface rule of §12.2 incl. "
         "terrain edits), the three integer samples of the containing triangle [i, k, height units of 1/1024 m] and "
         "edited (one of them has an edit). Changes nothing.",
         S::ObjectSchema({{"space", S::IntegerProperty("Space (default 0)")},
                          {"x", S::NumberProperty("x in metres")},
                          {"z", S::NumberProperty("z in metres")}},
                         {"x", "z"})},
        {"terrain_edits",
         "Reads the chunks of a space with terrain modification state (sequence > 0) in chunk order: chunk [x, z], "
         "sequence, has_deltas, non_zero_samples and the delta min / max in metres; paged (limit, default 100; after: "
         "the next value of the previous page). Changes nothing.",
         S::ObjectSchema({{"space", S::IntegerProperty("Space (default 0)")},
                          {"limit", S::IntegerProperty("Chunks per page (1..1000, default 100)")},
                          {"after", S::IntegerProperty("The next value of the previous page")}},
                         {})},
        {"chunk_info",
         "Reads one chunk's generated content state: placement_version (its record's, else the current one), "
         "recorded, stamped (host), sequence (the modification sequence of terrain edits and tombstones), tombstones "
         "[{slot, counter}] (at most 200, truncated), tombstone_count, instances per slot (the accepted tier-1 "
         "instances of the placement), the site keys of the chunk, the content_hash (generated on demand) and its "
         "residency on this peer {state none / requested / resident, parts, requested}. The chunk by chunk [x, z] or "
         "by "
         "position [x, z] in metres. Changes nothing.",
         S::ObjectSchema(
             {{"space", S::IntegerProperty("Space (default 0)")},
              {"chunk", chunkProperty("Chunk [x, z]")},
              {"position", positionXzProperty("A position [x, z] or [x, y, z] in metres (instead of chunk)")}},
             {})},
        {"streaming_status",
         "Reads the streaming state of this peer's world (WorldConfig::streaming; enabled false otherwise): "
         "synchronous, the statistics (requested, applied, results dropped, applied after a modification, unloaded, "
         "built inline, streaming_stalls (steps with a player on a chunk that is not ready), player stall steps, "
         "recent "
         "stalls (of the last 600 steps), island limit steps, mover holds, Simulated evaluations, resident bytes, the "
         "last streaming / simulation / load rebuild times in ms), per space (optional filter space) the requested, "
         "resident and pending chunks and the centres {entity, position, arrival}, the islands {count, limit, "
         "hard_limit}, the simulated entity count and the streaming pool {created, queued, running, residency_jobs}; "
         "with chunks: true the chunks in ChunkId order {space, chunk [x, z], state none / requested / resident, "
         "parts, "
         "wanted, requested, distance to the nearest centre, job}, paged (limit, default 100, at most 1000; after: the "
         "next value of the previous page). Changes nothing.",
         S::ObjectSchema({{"space", S::IntegerProperty("Only this space")},
                          {"chunks", Json::object{{"type", "boolean"}, {"description", "List the chunks"}}},
                          {"limit", S::IntegerProperty("Chunks per page (1..1000, default 100)")},
                          {"after", S::IntegerProperty("The next value of the previous page")}},
                         {})},
        {"save_status",
         "Reads the save directory of the world (S25; open false without one: --save <dir>): path, committed (the "
         "newest successful commit), in_flight (the commit being written, 0: none), last_commit_step and "
         "last_commit_time (world time of the last freeze), last_error (of the last failed commit), dirty_chunks "
         "(differing from the newest commit), stored_chunks / stored_groups (entity groups of unloaded chunks), "
         "unloaded_spaces, region_files, region_bytes, live_bytes, terrain_memory_bytes, and the statistics "
         "(groups and entities unloaded and reloaded, dropped references, failed groups, reload hook errors, spaces "
         "unloaded and reloaded, records read and written, bytes written, commits, failed commits, compactions, the "
         "last and largest freeze in ms, the last commit's write time in ms). Changes nothing.",
         S::ObjectSchema({}, {})},
        {"commit_save",
         "Commits the world to its save directory now (S25): freezes the state between steps and writes it on the IO "
         "lane (the commit point is the world-state file's rename). Returns {commit, deferred: false}; while a commit "
         "is in flight {deferred: true, in_flight} (the request commits at the first step boundary after it "
         "completed). An error without a save directory (--save <dir>) and on a client.",
         S::ObjectSchema({}, {})},
        {"list_sites",
         "Reads the sites (authored places) of a space's world plan in plan order: key, prefab, version, position "
         "[x, base height, z] in metres, quarter_turns, base_height, footprint [x0, z0, x1, z1] in local metres, the "
         "linked space IDs (the spaces the site created) and its chunks with their stamped state (host). Filter near "
         "{position, radius}; paged (limit, default 100, at most 1000; after: the next value of the previous page). "
         "Changes nothing.",
         S::ObjectSchema({{"space", S::IntegerProperty("Space (default 0)")},
                          {"near", Json::object{{"type", "object"},
                                                {"description", "{position: [x, z] or [x, y, z], radius} in metres"}}},
                          {"limit", S::IntegerProperty("Sites per page (1..1000, default 100)")},
                          {"after", S::IntegerProperty("The next value of the previous page")}},
                         {})},
        {"tombstone",
         "Removes one generated tier-1 instance (host only, between steps; the tombstone is replicated to the "
         "clients and saved; the instance disappears on every peer): chunk [x, z], slot by name (describe_world lists "
         "the generators' slots), counter (the candidate index). Changes the chunk's record and its sequence. Refused "
         "on clients, for an unknown slot, a counter out of range, a rejected or excluded candidate and a chunk "
         "outside "
         "the space. Idempotent: changed false for a tombstoned instance. Returns changed, chunk and the sequence.",
         S::ObjectSchema({{"space", S::IntegerProperty("Space (default 0)")},
                          {"chunk", chunkProperty("Chunk [x, z]")},
                          {"slot", S::StringProperty("Slot name of the space's generator")},
                          {"counter", S::IntegerProperty("Candidate index in the slot")}},
                         {"chunk", "slot", "counter"})},
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
         "Queues a one-shot player action (its name or game-defined code, optional aim [x,y,z]); delivered as "
         "ActionEvent in the next step, in (peer, sequence) order, exactly once.",
         S::ObjectSchema({{"player", entityProperty("The player entity")},
                          {"action", Json::object{{"type", Json::array{"integer", "string"}},
                                                  {"description", "Action name (describe_world lists them) or code"}}},
                          {"aim", S::Vec3Property("Aim point")}},
                         {"player", "action"})},
        {"pause", "Pauses the simulation (frames continue; with --coop the host and the client).", S::ObjectSchema({})},
        {"resume", "Resumes the simulation (with --coop the host and the client).", S::ObjectSchema({})},
        {"step",
         "Runs exactly n steps (default 1, at most 100000); only while paused (with --coop host and client "
         "interleaved, one step each). Returns the step index.",
         S::ObjectSchema({{"n", S::IntegerProperty("Number of steps")}})},
        {"time_scale", "Sets the simulation speed (0 stops it, 1 real time; with --coop of both worlds).",
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
         "Reads the step profile: per scope (step, phases, systems, command flushes) avg/min/p50/p95/p99/max and the "
         "mean of the largest 1% (worst_1pct_avg) in ms and "
         "the share of the parent, plus counters (entities, game objects, objects updated, commands, structural "
         "records, grid items), over the last steps (default: the whole history of 600). In the app also frame "
         "(frame and its poll, advance and render scopes; fps, steps per frame, dropped steps) and gpu (the GPU frame "
         "and its "
         "sections, from timestamps) over the last frames; with --coop also host: the hidden host world's step "
         "profile (the client runs no game systems). Changes nothing.",
         S::ObjectSchema(
             {{"steps", S::IntegerProperty("Window in steps (default all kept)")},
              {"frames", S::IntegerProperty("Window of the frame and GPU profiles in frames (default all)")},
              {"scopes", arrayProperty("Only scopes with these path prefixes (e.g. step/phase:behaviour)",
                                       Json::object{{"type", "string"}})}})},
        {"reset_profile", "Drops the profile history (in the app also the frame and GPU profiles).",
         S::ObjectSchema({})},
        {"logs",
         "Reads the engine log lines after a sequence number (at most the newest 500), optionally of one "
         "category (\"api\" lists the changes made through these tools; \"mcp\", the log of the tool calls, is only "
         "returned when asked for). Each line has seq, time (s since the start), category and message. Changes "
         "nothing.",
         S::ObjectSchema({{"since", S::IntegerProperty("Sequence number (0: all kept)")},
                          {"category", S::StringProperty("Category filter")}})},
        {"save_snapshot",
         "Saves the whole world state under a name in memory (at most 16 snapshots, 256 MB). Use it before an "
         "experiment: diff_snapshot shows what changed, load_snapshot rolls back (the snapshot tools are refused in "
         "--coop: the tools see the client world). Changes nothing in the world.",
         S::ObjectSchema({{"name", S::StringProperty("Snapshot name")},
                          {"overwrite", S::BooleanProperty("Replace an existing snapshot (false)")}},
                         {"name"})},
        {"load_snapshot",
         "Replaces the live world with a snapshot (game objects are recreated: configure and onLoad run). Entity "
         "handles from before the snapshot stay valid; ones obtained after it may name other entities now. "
         "changes_since reports one world_loaded entry. paused: set the paused state after the load. In the engine "
         "app a load neutralizes the local player's input (device input is transient).",
         S::ObjectSchema({{"name", S::StringProperty("Snapshot name")},
                          {"paused", S::BooleanProperty("Paused after the load (default: unchanged)")}},
                         {"name"})},
        {"list_snapshots", "Lists the snapshots (name, bytes, step, hash). Changes nothing.", S::ObjectSchema({})},
        {"delete_snapshot", "Deletes a snapshot (frees its memory). Changes nothing in the world.",
         S::ObjectSchema({{"name", S::StringProperty("Snapshot name")}}, {"name"})},
        {"diff_snapshot",
         "Compares a snapshot (a) with the live world or another snapshot (b, against): field differences per entity "
         "and component, entities and components only on one side (with class and position), and world-level "
         "differences (step, rng, events, actions). summary counts them per kind and component. Paged: limit "
         "(default 20, at most 1000), after (from next). Filters: entity, components. Changes nothing.",
         S::ObjectSchema({{"name", S::StringProperty("Snapshot name (a)")},
                          {"against", S::StringProperty("Another snapshot (b; default: the live world)")},
                          {"entity", entityProperty("Only this entity")},
                          {"components", arrayProperty("Only these components", Json::object{{"type", "string"}})},
                          {"limit", S::IntegerProperty("Maximum differences (1..1000, 20)")},
                          {"after", S::IntegerProperty("Skip this many differences (the next of the last call)")}},
                         {"name"})},
        {"export_snapshot",
         "Writes a snapshot to a file in the app's snapshot directory (a plain file name), so it survives a restart "
         "or a rebuild (to test migrations of an older save). An existing file is kept unless overwrite is true.",
         S::ObjectSchema({{"name", S::StringProperty("Snapshot name")},
                          {"file", S::StringProperty("File name")},
                          {"overwrite", S::BooleanProperty("Replace an existing file (false)")}},
                         {"name", "file"})},
        {"import_snapshot",
         "Reads a snapshot file from the app's snapshot directory into memory under a name (load_snapshot then "
         "loads it; an older save is migrated). Its hash is unknown (null) until it is loaded.",
         S::ObjectSchema({{"name", S::StringProperty("Snapshot name")},
                          {"file", S::StringProperty("File name")},
                          {"overwrite", S::BooleanProperty("Replace an existing snapshot (false)")}},
                         {"name", "file"})},
        {"get_data",
         "Reads the game's data files (tunable values, describe_world lists the types): with name one entry's "
         "values, else the entry names. Changes nothing.",
         S::ObjectSchema({{"type", S::StringProperty("Data type name")}, {"name", S::StringProperty("Entry name")}},
                         {"type"})},
        {"reload_data",
         "Re-reads the data files after you edited them (nothing changes if one is invalid or an entry was removed; "
         "with --coop the store both worlds share, reconfigured through the host, whose load resyncs the client). "
         "Values read through DataRefs change at once. With reconfigure (default true) the world is saved and loaded "
         "so every game object runs configure() again: derived (unsaved) state is rebuilt, changes_since reports a "
         "world_loaded entry (cursors answer reloaded: true, move tracking restarts) and a running "
         "measure_performance fails. Returns the types, the entry count and hook errors.",
         S::ObjectSchema({{"reconfigure", S::BooleanProperty("Reconfigure the game objects (true)")}})},
        {"apply",
         "Runs a batch: operations [{op: tool name, ...arguments}] or [{op: tool name, args: {arguments}}] (needed "
         "for edit_terrain, whose brush argument is named op too) in order, one result each. Stops at the first "
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
            // The shortest round-trip form, independent of the process locale (a comma decimal separator would split
            // a number into two array elements)
            const auto r = std::to_chars(buf, buf + sizeof(buf), v);
            out.append(buf, r.ptr);
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

// The profile of the last n samples of a measurement; n = 0 is an empty window (Profiler::report(0) means the whole
// history)
Json measurementReport(const Profiler& p, uint64_t n)
{
    if (n > 0) return p.report(int(n));
    return Json::object{
        {"unit", "ms"}, {"samples", 0}, {"top", Json::array{}}, {"scopes", Json::array{}}, {"counters", Json::array{}}};
}

// The app's profiles (frame, GPU) over the samples since a measurement started
std::vector<uint64_t> extraStarts(const WorldApi& api)
{
    std::vector<uint64_t> starts;
    for (const auto& p : api.extraProfiles()) starts.push_back(p.second->sampleCount());
    return starts;
}
Json withExtras(const Json& report, const WorldApi& api, const std::vector<uint64_t>& starts)
{
    Json::object o       = report.object_items();
    const auto& profiles = api.extraProfiles();
    for (size_t i = 0; i < profiles.size() && i < starts.size(); ++i)
        o[profiles[i].first] = measurementReport(*profiles[i].second, profiles[i].second->sampleCount() - starts[i]);
    return o;
}

// Co-op (S16b): the host's step profile over the steps it ran since start (the client runs no game systems)
Json withHost(const Json& report, const World* host, uint64_t start)
{
    if (!host) return report;
    Json::object o = report.object_items();
    o["host"]      = measurementReport(host->profiler(), host->profiler().sampleCount() - start);
    return o;
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

std::chrono::steady_clock::time_point answerDeadline(const McpServer& server)
{
    return server.callDeadline() -
           std::chrono::duration_cast<std::chrono::steady_clock::duration>(
               std::chrono::duration<double>(server.answerMarginFraction * server.toolTimeoutSeconds));
}

json11::Json profileWindow(const Profiler& profiler, uint64_t samples)
{
    return measurementReport(profiler, samples);
}

std::string exactToolJson(const json11::Json& j)
{
    return exactJson(j);
}

std::shared_ptr<WorldApi> registerWorldTools(McpServer& server, World& world, const WorldToolsOptions& options)
{
    auto api = std::make_shared<WorldApi>(world, options.allowEdits);
    api->setSnapshotDirectory(options.snapshotDirectory);
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
                       [weak, op, serverPtr = &server](const Json& args, McpServer::Respond respond)
                       {
                           auto a = weak.lock();
                           if (!a) return respond(ToolResult::Error("the world is not available"));
                           // The request's real deadline (from when it was queued) minus a margin for the answer
                           a->setDeadline(answerDeadline(*serverPtr));
                           respond(toResult(a->call(op, args)));
                       });
    }

    // Runs while paused (exactly n steps, within a wall-clock budget), or waits for n steps while running (answered
    // before the tool call times out, with the steps measured so far)
    server.addTool(
        "measure_performance",
        "Measures n steps (default 120, at most 600: the profile history) and returns the step profile of exactly "
        "those "
        "steps (see get_profile). Paused: runs the steps now (a pure simulation benchmark; edits must be allowed). "
        "Running: waits until the app has run them (refused at time scale 0); it answers early with fewer steps if "
        "they take too long; in the app the result also has the frame and gpu profiles of those frames. Changes "
        "nothing but time while paused. With --coop paused steps run host and client interleaved and the result has "
        "host: the host's step profile of the same steps. frames: n (1..600, instead of steps; the app with a device) "
        "measures the next "
        "n frame loops instead, also while paused (a render-only measurement): the frame profile of those loops, the "
        "gpu profile of the frames they rendered and the step profile of the steps they ran.",
        [&]()
        {
            Json::object schema =
                McpServer::ObjectSchema(
                    {{"steps", McpServer::IntegerProperty("Number of steps (1..600)")},
                     {"frames", McpServer::IntegerProperty("Number of frame loops (1..600), instead of steps")}})
                    .object_items();
            schema["additionalProperties"] = false;
            return Json(schema);
        }(),
        [weak, serverPtr = &server, measureFrames = options.measureFrames](const Json& args, McpServer::Respond respond)
        {
            // The request's deadline (from when it was queued, minus a margin for the answer)
            const auto deadline = answerDeadline(*serverPtr);
            auto a              = weak.lock();
            if (!a || !a->world()) return respond(ToolResult::Error("the world is not available"));
            for (auto& kv : args.object_items())
                if (kv.first != "steps" && kv.first != "frames")
                    return respond(ToolResult::Error("unknown argument \"" + kv.first + "\""));
            if (!args["frames"].is_null())
            {
                if (!args["steps"].is_null())
                    return respond(ToolResult::Error("measure_performance: steps or frames, not both"));
                const double f = args["frames"].number_value();
                if (!args["frames"].is_number() || !(f >= 1 && f <= 600) || f != std::floor(f))
                    return respond(ToolResult::Error("frames: an integer 1..600"));
                if (!measureFrames)
                    return respond(
                        ToolResult::Error("measure_performance: this app renders no frames (frames needs "
                                          "an app with a device)"));
                return measureFrames(int(f), deadline, respond);
            }
            World& w       = *a->world();
            const double n = args["steps"].is_null() ? 120 : args["steps"].number_value();
            const int max  = w.profiler().historySize();
            if ((!args["steps"].is_null() && !args["steps"].is_number()) || !(n >= 1 && n <= max) || n != std::floor(n))
                return respond(ToolResult::Error("steps: an integer 1.." + std::to_string(max)));
            if (w.inStep()) return respond(ToolResult::Error("measure_performance: only between steps"));
            auto over = [deadline]() { return std::chrono::steady_clock::now() >= deadline; };
            if (w.isPaused())
            {
                if (!a->editsAllowed())
                    return respond(ToolResult::Error("measure_performance: running steps while paused is an edit"));
                int ran                            = 0;
                const std::vector<uint64_t> starts = extraStarts(*a);
                World* host                        = a->inCoop() ? a->coop().host : nullptr;
                const uint64_t hostStart           = host ? host->profiler().sampleCount() : 0;
                while (ran < int(n) && !over())
                {
                    // Co-op (S16b): host one step, then the client one step
                    if (host && a->coop().stepAll)
                        a->coop().stepAll(1);
                    else
                        w.step();
                    ran++;
                }
                Json report =
                    withHost(withExtras(measurementReport(w.profiler(), uint64_t(ran)), *a, starts), host, hostStart);
                if (ran < int(n))
                {
                    Json::object o     = report.object_items();
                    o["stopped_early"] = true;
                    o["requested"]     = n;
                    report             = o;
                }
                return respond(ToolResult::Text(exactJson(report)));
            }
            if (w.timeScale() == 0)
                return respond(ToolResult::Error("measure_performance: the time scale is 0 (no steps run)"));
            // Answered exactly once: after n steps, at the deadline (checked after every step and every poll, so also
            // when the world stops stepping), when the profile is reset, or when the world is destroyed
            struct Measurement
            {
                bool answered  = false;
                bool worldGone = false;
                uint64_t start, resets, loads;
                std::vector<uint64_t> extras;
                World* host         = nullptr;  // co-op (S16b): the host's profile over the same time
                uint64_t hostStart  = 0;
                uint64_t hostResets = 0, hostLoads = 0;
                World* world        = nullptr;  // holds the destroy and step listeners
                int destroyListener = -1;
                int stepListener    = -1;
            };
            auto m             = std::make_shared<Measurement>();
            m->start           = w.profiler().sampleCount();
            m->resets          = w.profiler().resetCount();
            m->loads           = w.loadCount();
            m->extras          = extraStarts(*a);
            m->host            = a->inCoop() ? a->coop().host : nullptr;
            m->hostStart       = m->host ? m->host->profiler().sampleCount() : 0;
            m->hostResets      = m->host ? m->host->profiler().resetCount() : 0;
            m->hostLoads       = m->host ? m->host->loadCount() : 0;
            m->world           = &w;
            m->destroyListener = w.addDestroyListener(
                [respond, m]()
                {
                    m->worldGone = true;
                    if (m->answered) return;
                    m->answered = true;
                    respond(ToolResult::Error("the world was destroyed during the measurement"));
                });
            // Returns whether the measurement still runs
            auto check = [weak, respond, n, over, m]()
            {
                if (m->answered) return false;
                auto api = weak.lock();
                if (!api || !api->world())
                {
                    m->answered = true;
                    if (!m->worldGone)
                    {
                        m->world->removeDestroyListener(m->destroyListener);
                        m->world->removeStepListener(m->stepListener);
                    }
                    respond(ToolResult::Error("the world is not available"));
                    return false;
                }
                World& world = *api->world();
                if (&world != m->world)
                {
                    // The API moved to another world (rebind)
                    m->answered = true;
                    if (!m->worldGone)
                    {
                        m->world->removeDestroyListener(m->destroyListener);
                        m->world->removeStepListener(m->stepListener);
                    }
                    respond(ToolResult::Error("the world was replaced during the measurement"));
                    return false;
                }
                // Co-op (local review): a load or profile reset of the host counts too (reload_data loads the host)
                const bool hostLoaded = m->host && m->host->loadCount() != m->hostLoads;
                const bool hostReset  = m->host && m->host->profiler().resetCount() != m->hostResets;
                if (world.profiler().resetCount() != m->resets || world.loadCount() != m->loads || hostLoaded ||
                    hostReset)
                {
                    m->answered = true;
                    world.removeDestroyListener(m->destroyListener);
                    world.removeStepListener(m->stepListener);
                    respond(ToolResult::Error(world.loadCount() != m->loads || hostLoaded
                                                  ? "the world was loaded during the measurement"
                                                  : "the profile was reset during the measurement"));
                    return false;
                }
                const uint64_t done = world.profiler().sampleCount() - m->start;
                if (done < uint64_t(n) && !over()) return true;
                m->answered = true;
                world.removeDestroyListener(m->destroyListener);
                world.removeStepListener(m->stepListener);
                // No step ran: an empty window (report(0) would be the whole history)
                Json report = withHost(withExtras(measurementReport(world.profiler(), done), *api, m->extras), m->host,
                                       m->hostStart);
                if (done < uint64_t(n))
                {
                    Json::object o     = report.object_items();
                    o["stopped_early"] = true;
                    o["requested"]     = n;
                    report             = o;
                }
                respond(ToolResult::Text(exactJson(report)));
                return false;
            };
            m->stepListener = w.addStepListener(check);
            serverPtr->addPollListener(check);
        });
    return api;
}

}  // namespace Engine
}  // namespace Saiga
