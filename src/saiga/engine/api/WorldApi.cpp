/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// WorldApi: the world as JSON by names (list, inspect, change, spawn, destroy, events, actions, time, changes,
// profile, logs).

#include "saiga/engine/api/WorldApi.h"

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/Systems.h"
#include "saiga/engine/save/SaveDiff.h"
#include "saiga/engine/save/SaveFormat.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace Saiga
{
namespace Engine
{
using Json = json11::Json;

namespace
{
constexpr size_t kJournalCapacity = 100000;

struct Operation
{
    const char* name;
    bool edit;
    std::vector<const char*> arguments;
};
const std::vector<Operation>& operationTable()
{
    static const std::vector<Operation> table = {
        {"describe_world", false, {}},
        {"world_status", false, {}},
        {"list_entities", false, {"class", "components", "near", "include", "limit", "after"}},
        {"inspect", false, {"entity", "components"}},
        {"get_global", false, {"component", "space"}},
        {"changes_since", false, {"cursor", "include_moved", "limit"}},
        {"get_profile", false, {"steps", "scopes"}},
        {"logs", false, {"since", "category"}},
        {"save_snapshot", false, {"name", "overwrite"}},
        {"list_snapshots", false, {}},
        {"delete_snapshot", false, {"name"}},
        {"diff_snapshot", false, {"name", "against", "entity", "components", "limit", "after"}},
        {"get_data", false, {"type", "name"}},
        {"load_snapshot", true, {"name", "paused"}},
        {"export_snapshot", true, {"name", "file", "overwrite"}},
        {"import_snapshot", true, {"name", "file", "overwrite"}},
        {"reload_data", true, {"reconfigure"}},
        {"set_component", true, {"entity", "component", "values"}},
        {"add_component", true, {"entity", "component", "values"}},
        {"remove_component", true, {"entity", "component"}},
        {"set_transform", true, {"entity", "position", "rotation", "scale", "teleport"}},
        {"spawn", true, {"class", "position", "rotation", "scale", "space", "components"}},
        {"destroy", true, {"entity"}},
        {"set_global", true, {"component", "values", "space"}},
        {"emit_event", true, {"event", "values"}},
        {"queue_action", true, {"player", "action", "aim"}},
        {"pause", true, {}},
        {"resume", true, {}},
        {"step", true, {"n"}},
        {"time_scale", true, {"scale"}},
        {"reset_profile", true, {}},
        {"apply", true, {"operations"}},
    };
    return table;
}
const Operation* findOperation(const std::string& name)
{
    for (const Operation& o : operationTable())
        if (name == o.name) return &o;
    return nullptr;
}

const Json& emptyObject()
{
    static const Json empty = Json::object{};
    return empty;
}

// An integer argument in [min, max] (absent: def); rejects strings, fractions and out-of-range values
bool intArg(const Json& a, const char* key, int64_t min, int64_t max, int64_t def, int64_t& out, std::string& error)
{
    const Json& j = a[key];
    if (j.is_null())
    {
        out = def;
        return true;
    }
    const double d = j.number_value();
    if (!j.is_number() || d != std::floor(d) || !(d >= double(min) && d <= double(max)))
    {
        error = std::string(key) + ": an integer in " + std::to_string(min) + ".." + std::to_string(max);
        return false;
    }
    out = int64_t(d);
    return true;
}

bool boolArg(const Json& a, const char* key, bool def, bool& out, std::string& error)
{
    const Json& j = a[key];
    if (j.is_null())
    {
        out = def;
        return true;
    }
    if (!j.is_bool())
    {
        error = std::string(key) + ": expected true or false";
        return false;
    }
    out = j.bool_value();
    return true;
}

bool readVec3(const Json& j, Vec3& out, std::string& error, const char* what)
{
    JsonReader r(emptyObject(), true);
    Vec3 v = out;
    r.read(j, v, what);
    if (!r.ok())
    {
        error = r.errorMessage();
        return false;
    }
    if (!std::isfinite(v.x()) || !std::isfinite(v.y()) || !std::isfinite(v.z()))
    {
        error = std::string(what) + ": not finite";
        return false;
    }
    out = v;
    return true;
}

std::string truncateForLog(const std::string& s)
{
    return s.size() <= 300 ? s : s.substr(0, 300) + "...";
}

bool stateEntity(const Registry& reg, Entity e)
{
    return reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState);
}

// An alive entity from the argument
bool aliveEntity(World& w, const Json& j, Entity& e, std::string& error)
{
    if (j.is_null())
    {
        error = "entity: missing";
        return false;
    }
    if (!WorldApi::parseEntity(j, e, error)) return false;
    if (!w.alive(e))
    {
        error = "entity " + std::to_string(e.index) + ":" + std::to_string(e.generation) + " is not alive";
        return false;
    }
    return true;
}

// A registered component by name
bool componentId(const Registry& reg, const Json& j, ComponentId& id, std::string& error)
{
    if (!j.is_string())
    {
        error = "component: expected a component name";
        return false;
    }
    const int found = reg.findComponent(j.string_value());
    if (found < 0)
    {
        error = "unknown component \"" + j.string_value() + "\" (describe_world lists them)";
        return false;
    }
    id = ComponentId(found);
    return true;
}

Json componentJson(World& w, Entity e, ComponentId id)
{
    if (id == Transform::id) return toJson(w.read<Transform>(e));
    return w.registry().poolById(id)->toJson(e.index);
}

const char* className(const World& w, Entity e)
{
    const Behaviour* b = w.tryRead<Behaviour>(e);
    if (!b) return nullptr;
    const ClassList::ClassInfo* c = w.classList().find(b->classId);
    return c ? c->name : nullptr;
}

// The direct children (Parent pool scan, index order)
std::vector<Entity> childrenOf(World& w, Entity parent)
{
    std::vector<Entity> out;
    w.viewAll<const Parent>().each(
        [&](Entity c, const Parent& p)
        {
            if (p.entity == parent) out.push_back(c);
        });
    std::sort(out.begin(), out.end());
    return out;
}

// p is e or one of e's descendants (a Parent link from e to p would close a cycle)
bool isSelfOrDescendant(World& w, Entity e, Entity p)
{
    const uint32_t limit = w.registry().slotCount();
    for (uint32_t n = 0; n <= limit && p.valid(); ++n)
    {
        if (p == e) return true;
        const Parent* link = w.tryRead<Parent>(p);
        p                  = link ? link->entity : Entity{};
    }
    return false;
}

// A Parent value from JSON, checked: an alive parent, no cycle, no world-state child (child may be invalid: a new
// entity of spawn)
bool parentFromJson(World& w, Entity child, const Json& values, Parent& out, std::string& error)
{
    Parent p = out;
    if (!fromJson(values, p, &error, true)) return false;
    if (!w.alive(p.entity))
    {
        error = "Parent: the parent entity is not alive";
        return false;
    }
    if (stateEntity(w.registry(), p.entity))
    {
        error = "Parent: a world-state entity is not a parent";
        return false;
    }
    if (child.valid() && stateEntity(w.registry(), child))
    {
        error = "Parent: world-state entities cannot have a parent";
        return false;
    }
    if (child.valid() && isSelfOrDescendant(w, child, p.entity))
    {
        error = "Parent: the link would form a cycle";
        return false;
    }
    out = p;
    return true;
}

// Components that the generic add/remove/spawn paths do not handle
std::string specialComponent(ComponentId id, bool adding)
{
    if (id == Transform::id) return "Transform: every entity has one (use set_transform)";
    if (id == Behaviour::id && adding) return "Behaviour: added by spawn with a class";
    if (id == Persistent::id && adding) return "Persistent: assigned by makePersistent";
    return {};
}

// Checks the values of an initial or added component (nothing is changed)
bool checkComponentValues(World& w, Entity child, ComponentId id, const Json& values, std::string& error)
{
    const std::string special = specialComponent(id, true);
    if (!special.empty())
    {
        error = special;
        return false;
    }
    if (id == Parent::id)
    {
        Parent p;
        return parentFromJson(w, child, values.is_null() ? emptyObject() : values, p, error);
    }
    if (values.is_null()) return true;
    return w.registry().poolById(id)->checkJson(values, error);
}

// Adds a component with values (checked before; a Parent is checked again: hooks may have changed the world since)
bool addComponentValues(World& w, Entity e, ComponentId id, const Json& values, std::string& error)
{
    if (id == Parent::id)
    {
        Parent p;
        if (!parentFromJson(w, e, values.is_null() ? emptyObject() : values, p, error)) return false;
        w.add<Parent>(e, p);
        return true;
    }
    w.addDefaultNow(e, id);
    if (!values.is_null()) return w.registry().poolById(id)->mergeJson(e.index, values, error);
    return true;
}

// The fields set_component accepts back (an undo payload): Transform without space, Behaviour without the derived and
// fixed fields
Json undoJson(ComponentId id, const Json& values)
{
    if (id != Transform::id && id != Behaviour::id) return values;
    Json::object o = values.object_items();
    o.erase("space");
    o.erase("tickInterval");
    o.erase("classId");
    return o;
}

// The opaque cursor of changes_since: "<epoch>.<seq>.<from>.<to>.<index>": the session (a random epoch per API, so a
// cursor of another session is recognized), the last journal sequence number read, and the move window: moves after
// step `from` are reported; while a window is paged, `to` (> 0) is its last step and `index` the entity index after
// which it continues. Structural paging and move paging are independent.
struct Cursor
{
    uint32_t epoch = 0;
    uint64_t seq = 0, from = 0, to = 0;
    uint32_t index = 0;
};
std::string makeCursor(const Cursor& c)
{
    return std::to_string(c.epoch) + "." + std::to_string(c.seq) + "." + std::to_string(c.from) + "." +
           std::to_string(c.to) + "." + std::to_string(c.index);
}
bool parseCursor(const std::string& s, Cursor& c)
{
    uint64_t parts[5];
    const char* p = s.c_str();
    for (int k = 0; k < 5; ++k)
    {
        char* end = nullptr;
        if (*p < '0' || *p > '9') return false;
        parts[k] = std::strtoull(p, &end, 10);
        if (k < 4 && *end != '.') return false;
        if (k == 4 && *end != 0) return false;
        p = end + 1;
    }
    if (parts[0] > 0xFFFFFFFFull || parts[4] > 0xFFFFFFFFull) return false;
    c = Cursor{uint32_t(parts[0]), parts[1], parts[2], parts[3], uint32_t(parts[4])};
    return true;
}
}  // namespace

// ---- setup --------------------------------------------------------------------------------------------------------

WorldApi::WorldApi(World& world, bool allowEdits)
    : w(&world), allowEdits(allowEdits), alive(std::make_shared<bool>(true))
{
    // A random session epoch: cursors of other API instances (another session, a restarted game) are rejected
    try
    {
        std::random_device rd;
        epoch = rd() % 1000000000u + 1;
    }
    catch (...)
    {
        // random_device may be unavailable (sandboxes): the clock is unique enough for a session epoch
        epoch = uint32_t(std::chrono::steady_clock::now().time_since_epoch().count() % 1000000000) + 1;
    }
    attach();
}

void WorldApi::attach()
{
    consumer                 = w->structuralLog().addConsumer();
    std::weak_ptr<bool> weak = alive;
    // The journal is filled after every step, so the structural log is never held back by an idle API
    stepListener = w->addStepListener(
        [this, weak]()
        {
            if (weak.expired()) return false;
            drainChanges();
            if (movesTracked) trackMoves();
            return true;
        });
    destroyListener = w->addDestroyListener(
        [this, weak]()
        {
            if (!weak.expired()) w = nullptr;
        });
    loadListener = w->addLoadListener(
        [this, weak](const LoadInfo& info)
        {
            if (!weak.expired()) onLoaded(info);
        });
    if (movesTracked) w->retainStepChanges();
}

void WorldApi::detach()
{
    if (!w) return;
    w->structuralLog().removeConsumer(consumer);
    w->removeDestroyListener(destroyListener);
    w->removeStepListener(stepListener);
    w->removeLoadListener(loadListener);
    if (movesTracked) w->releaseStepChanges();
}

WorldApi::~WorldApi()
{
    alive.reset();
    detach();
}

void WorldApi::markLoaded(uint64_t step, uint64_t fromStep)
{
    // One journal entry for a load or a world swap; moves start again
    generation++;
    lastDestroyed = Entity{};
    journal.push_back(Change{nextSeq++, step, "world_loaded", Entity{}, -1, fromStep, generation});
    if (journal.size() > kJournalCapacity) journal.pop_front();
    lastLoadSeq = journal.back().seq;
    lastMoved.clear();
    windows.clear();
    movesSince = step;
}

void WorldApi::journalRecord(const StructuralLog::Record& r, uint64_t stepIndex)
{
    if (r.component == Transform::id)
    {
        // Destroy appends Transform first, then the other components: folded
        lastDestroyed = r.added ? Entity{} : r.entity;
        journal.push_back(Change{nextSeq++, stepIndex, r.added ? "created" : "destroyed", r.entity, -1});
    }
    else if (!r.added && r.entity == lastDestroyed)
        return;
    else
    {
        lastDestroyed = Entity{};
        journal.push_back(Change{nextSeq++, stepIndex, r.added ? "added" : "removed", r.entity, int(r.component)});
    }
    if (journal.size() > kJournalCapacity) journal.pop_front();
}

void WorldApi::rebind(World& world)
{
    const uint64_t fromStep = w ? w->stepIndex() : 0;
    if (w) drainChanges();
    detach();
    w = &world;
    attach();
    markLoaded(w->stepIndex(), fromStep);
}

void WorldApi::onLoaded(const LoadInfo& info)
{
    // The records before the load are journaled as usual, the load's own records [logBegin, logEnd) are folded into
    // one entry, records after it (e.g. made by other load listeners) are journaled again
    StructuralLog& log       = w->structuralLog();
    uint64_t index           = log.cursor(consumer);
    bool marked              = false;
    const uint64_t stepIndex = w->stepIndex();
    log.consume(consumer,
                [&](const StructuralLog::Record& r)
                {
                    const uint64_t at = index++;
                    // Records before the load happened at the step before it
                    if (at < info.logBegin) return journalRecord(r, info.fromStep);
                    if (at < info.logEnd) return;
                    if (!marked)
                    {
                        markLoaded(info.toStep, info.fromStep);
                        marked = true;
                    }
                    journalRecord(r, stepIndex);
                });
    if (!marked) markLoaded(info.toStep, info.fromStep);
}

std::vector<std::string> WorldApi::operations() const
{
    std::vector<std::string> names;
    for (const Operation& o : operationTable())
        if (allowEdits || !o.edit) names.push_back(o.name);
    return names;
}

std::vector<std::string> WorldApi::arguments(const std::string& operation)
{
    std::vector<std::string> names;
    if (const Operation* o = findOperation(operation))
        for (const char* n : o->arguments) names.push_back(n);
    return names;
}

bool WorldApi::isEdit(const std::string& operation)
{
    const Operation* o = findOperation(operation);
    return o && o->edit;
}

bool WorldApi::parseEntity(const Json& j, Entity& e, std::string& error)
{
    JsonReader r(emptyObject(), true);
    if (!r.parseEntity(j, e, "entity"))
    {
        error = r.errorMessage();
        return false;
    }
    return true;
}

Json WorldApi::entityJson(Entity e)
{
    return Json::object{{"index", double(e.index)}, {"generation", double(e.generation)}};
}

std::chrono::steady_clock::time_point WorldApi::budgetDeadline() const
{
    if (hasDeadline) return deadline;
    return std::chrono::steady_clock::now() +
           std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(budget));
}

ApiResult WorldApi::call(const std::string& operation, const Json& a)
{
    if (!w) return ApiResult::failure("the world is gone");
    const Json& args = a.is_null() ? emptyObject() : a;
    if (!args.is_object()) return ApiResult::failure("the arguments must be an object");
    const Operation* op = findOperation(operation);
    if (!op) return ApiResult::failure("unknown operation \"" + operation + "\"");
    // Unknown argument names are errors (a typo must not silently do nothing)
    for (auto& kv : args.object_items())
        if (std::find_if(op->arguments.begin(), op->arguments.end(), [&](const char* n) { return kv.first == n; }) ==
            op->arguments.end())
        {
            std::string expected;
            for (const char* n : op->arguments) expected += std::string(expected.empty() ? "" : ", ") + n;
            return ApiResult::failure(operation + ": unknown argument \"" + kv.first +
                                      "\" (expected: " + (expected.empty() ? "none" : expected) + ")");
        }
    // apply checks the permission per operation
    if (op->edit && !allowEdits && operation != "apply")
        return ApiResult::failure(operation + ": edits are not allowed (read-only)");
    if (w->inStep()) return ApiResult::failure(operation + ": only between steps");
    // A step listener must not change the world through the API (a step inside it is not supported)
    // (apply checks its operations before running any: a read-only batch is allowed)
    if (op->edit && w->dispatchingListeners() && operation != "apply")
        return ApiResult::failure(operation + ": not inside a step listener");
    ApiResult result;
    try
    {
        result = run(operation, args);
    }
    catch (const std::exception& ex)
    {
        // A game hook threw (e.g. onSpawn): the world cleaned up, but earlier side effects of the operation may remain
        result = ApiResult::failure(operation + ": a game hook threw: " + ex.what() +
                                    " (side effects before the exception may remain)");
    }
    catch (...)
    {
        result = ApiResult::failure(operation +
                                    ": a game hook threw an exception (side effects before the exception may remain)");
    }
    // Edits are logged after they ran, with their outcome (failed attempts are marked)
    if (op->edit && operation != "apply")
        SAIGA_LOG("api", "%s %s %s%s", result.ok ? "ok" : "failed:", operation.c_str(),
                  truncateForLog(args.dump()).c_str(), result.ok ? "" : (" -> " + result.error).c_str());
    return result;
}

ApiResult WorldApi::run(const std::string& operation, const Json& args)
{
    if (operation == "describe_world") return describeWorld(args);
    if (operation == "world_status") return worldStatus(args);
    if (operation == "list_entities") return listEntities(args);
    if (operation == "inspect") return inspect(args);
    if (operation == "get_global") return getGlobal(args);
    if (operation == "changes_since") return changesSince(args);
    if (operation == "get_profile") return getProfile(args);
    if (operation == "logs") return logs(args);
    if (operation == "set_component") return setComponent(args);
    if (operation == "add_component") return addComponent(args);
    if (operation == "remove_component") return removeComponent(args);
    if (operation == "set_transform") return setTransform(args);
    if (operation == "spawn") return spawn(args);
    if (operation == "destroy") return destroy(args);
    if (operation == "set_global") return setGlobal(args);
    if (operation == "emit_event") return emitEvent(args);
    if (operation == "queue_action") return queueAction(args);
    if (operation == "pause" || operation == "resume")
    {
        w->setPaused(operation == "pause");
        return ApiResult::success(Json::object{{"paused", w->isPaused()}, {"step", double(w->stepIndex())}});
    }
    if (operation == "step") return step(args);
    if (operation == "save_snapshot") return saveSnapshot(args);
    if (operation == "load_snapshot") return loadSnapshot(args);
    if (operation == "list_snapshots") return listSnapshots(args);
    if (operation == "delete_snapshot") return deleteSnapshot(args);
    if (operation == "diff_snapshot") return diffSnapshot(args);
    if (operation == "export_snapshot") return exportSnapshot(args);
    if (operation == "import_snapshot") return importSnapshot(args);
    if (operation == "get_data") return getData(args);
    if (operation == "reload_data") return reloadData(args);
    if (operation == "time_scale") return timeScale(args);
    if (operation == "reset_profile")
    {
        w->profiler().reset();
        return ApiResult::success(Json::object{{"reset", true}});
    }
    return apply(args);
}

// ---- reading ------------------------------------------------------------------------------------------------------

ApiResult WorldApi::describeWorld(const Json&)
{
    const Registry& reg = w->registry();
    auto fieldsJson     = [](const std::vector<FieldInfo>& fields)
    {
        Json::array a;
        for (const FieldInfo& f : fields)
        {
            Json::object o{{"name", f.name}, {"type", f.type}};
            if (!f.enumValues.empty())
            {
                Json::array values;
                for (auto& v : f.enumValues) values.push_back(v);
                o["values"] = values;
            }
            a.push_back(o);
        }
        return a;
    };
    static const char* kReplicate[] = {"never", "host", "host_reliable", "owner_authoritative", "cosmetic"};
    Json::array components;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!reg.registered(ComponentId(id))) continue;
        const Registry::ComponentInfo& info = reg.componentInfo(ComponentId(id));
        std::vector<FieldInfo> fields =
            id == Transform::id ? describeFields<Transform>() : reg.poolById(ComponentId(id))->fields();
        Json::object o{{"name", info.name},
                       {"id", int(id)},
                       {"replicate", kReplicate[int(info.replicate)]},
                       {"saved", info.saved == Saved::Yes},
                       {"fields", fieldsJson(fields)}};
        if (info.global) o["global"] = true;
        if (info.spaceState) o["space_state"] = true;
        if (id == Transform::id)
            o["note"] =
                "every entity has one; change it with set_transform (children follow their parent: change "
                "their LocalTransform)";
        if (id == Behaviour::id)
            o["note"] =
                "added by spawn with a class; classId is fixed; tickInterval is derived from the rings around "
                "the players (set alwaysTickEveryStep instead)";
        if (id == Persistent::id) o["note"] = "assigned by makePersistent; read-only";
        if (id == Parent::id) o["note"] = "{entity}: the parent must be alive and must not be a descendant";
        components.push_back(o);
    }
    Json::array classes;
    for (const auto& c : w->classList().all())
        if (c.factory) classes.push_back(Json::object{{"name", c.name}, {"id", int(c.id)}});
    Json::array events;
    for (int id = 0; id < 65536; ++id)
        if (const EventQueueBase* q = w->events.find(EventId(id)))
            events.push_back(Json::object{{"name", q->name}, {"id", id}, {"fields", fieldsJson(q->fields())}});
    Json::array dataTypes;
    for (const auto& t : w->dataTables().tables())
    {
        Json::array names;
        for (auto& n : t->names()) names.push_back(n);
        dataTypes.push_back(Json::object{
            {"name", t->name}, {"file", t->file}, {"fields", fieldsJson(t->fields())}, {"entries", names}});
    }
    Json::array ops;
    for (auto& o : operations()) ops.push_back(o);
    return ApiResult::success(Json::object{{"components", components},
                                           {"classes", classes},
                                           {"events", events},
                                           {"entity_format", "{\"index\": i, \"generation\": g} or \"i:g\""},
                                           {"operations", ops},
                                           {"data_types", dataTypes},
                                           {"save_format_version", double(kSaveFormatVersion)},
                                           {"edits_allowed", allowEdits}});
}

ApiResult WorldApi::worldStatus(const Json&)
{
    const Registry& reg  = w->registry();
    const size_t objects = reg.registered(Behaviour::id) ? reg.poolById(Behaviour::id)->size() : 0;
    char hash[32];
    std::snprintf(hash, sizeof(hash), "%016llx", (unsigned long long)w->hash());
    return ApiResult::success(Json::object{{"step", double(w->stepIndex())},
                                           {"time", w->time()},
                                           {"paused", w->isPaused()},
                                           {"time_scale", w->timeScale()},
                                           {"steps_per_second", w->worldConfig().stepsPerSecond},
                                           {"entities", double(reg.aliveCount())},
                                           {"game_objects", double(objects)},
                                           {"hash", hash},
                                           {"world_generation", double(generation)}});
}

ApiResult WorldApi::listEntities(const Json& a)
{
    const Registry& reg = w->registry();
    std::string error;
    ComponentMask mask;
    if (!a["components"].is_null())
    {
        if (!a["components"].is_array()) return ApiResult::failure("components: expected an array of names");
        for (const Json& n : a["components"].array_items())
        {
            ComponentId id;
            if (!componentId(reg, n, id, error)) return ApiResult::failure(error);
            mask.set(id);
        }
    }
    // include: the values of these components in each result (e.g. Health for "heal everything below 50")
    std::vector<ComponentId> include;
    if (!a["include"].is_null())
    {
        if (!a["include"].is_array()) return ApiResult::failure("include: expected an array of component names");
        for (const Json& n : a["include"].array_items())
        {
            ComponentId id;
            if (!componentId(reg, n, id, error)) return ApiResult::failure("include: " + error);
            include.push_back(id);
        }
    }
    int classFilter = -1;
    if (!a["class"].is_null())
    {
        if (!a["class"].is_string()) return ApiResult::failure("class: expected a class name");
        const ClassList::ClassInfo* c = w->classList().find(a["class"].string_value());
        if (!c) return ApiResult::failure("unknown class \"" + a["class"].string_value() + "\"");
        classFilter = c->id;
    }
    int64_t limit = 0, after = 0;
    if (!intArg(a, "limit", 1, 1000, 100, limit, error) || !intArg(a, "after", 0, 0xFFFFFFFFll, 0, after, error))
        return ApiResult::failure(error);

    // Candidates in index order: the grid for near (positions of the last step), else every slot
    std::vector<Entity> candidates;
    std::vector<std::pair<Entity, Vec3>> gridHits;  // near: in index order, like candidates
    Vec3 center       = Vec3::Zero();
    const bool isNear = !a["near"].is_null();
    if (isNear)
    {
        const Json& near = a["near"];
        if (!near.is_object()) return ApiResult::failure("near: expected {space?, position, radius}");
        for (auto& kv : near.object_items())
            if (kv.first != "space" && kv.first != "position" && kv.first != "radius")
                return ApiResult::failure("near: unknown argument \"" + kv.first + "\"");
        if (near["position"].is_null()) return ApiResult::failure("near.position: missing");
        if (!readVec3(near["position"], center, error, "near.position")) return ApiResult::failure(error);
        const double radius = near["radius"].number_value();
        if (!near["radius"].is_number() || !(radius >= 0) || !std::isfinite(radius))
            return ApiResult::failure("near.radius: a finite number >= 0");
        int64_t space = 0;
        if (!intArg(near, "space", 0, 65535, 0, space, error)) return ApiResult::failure("near." + error);
        // The grid's positions (the ones the selection used) for the distance
        w->forEachWithin(SpaceId(space), center, radius, ComponentMask(),
                         [&](Entity e, const Vec3& p) { gridHits.push_back({e, p}); });
        std::sort(gridHits.begin(), gridHits.end(),
                  [](const std::pair<Entity, Vec3>& x, const std::pair<Entity, Vec3>& y) { return x.first < y.first; });
        for (auto& h : gridHits) candidates.push_back(h.first);
    }
    else
    {
        for (uint32_t i = uint32_t(after) + 1; i < reg.slotCount(); ++i)
        {
            const Entity e = reg.aliveAt(i);
            if (e.valid()) candidates.push_back(e);
        }
    }

    Json::array out;
    uint32_t last = 0;
    bool more     = false;
    for (Entity e : candidates)
    {
        if (e.index <= uint32_t(after)) continue;
        if (!reg.entityMask(e).containsAll(mask)) continue;
        const Behaviour* b = w->tryRead<Behaviour>(e);
        if (classFilter >= 0 && (!b || b->classId != classFilter)) continue;
        if (out.size() == size_t(limit))
        {
            more = true;
            break;
        }
        Json::array names;
        const ComponentMask& m = reg.entityMask(e);
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) names.push_back(reg.componentInfo(ComponentId(id)).name);
        const Vec3 position = w->read<Transform>(e).position();
        Json::object o{{"entity", entityJson(e)}, {"position", toJson(position)}, {"components", names}};
        if (const char* c = className(*w, e)) o["class"] = c;
        if (isNear)
        {
            // The selection used the grid position of the last step (the live position may have moved since)
            auto hit           = std::lower_bound(gridHits.begin(), gridHits.end(), e,
                                                  [](const std::pair<Entity, Vec3>& x, Entity y) { return x.first < y; });
            o["grid_position"] = toJson(hit->second);
            o["distance"]      = (hit->second - center).norm();
        }
        if (!include.empty())
        {
            Json::object values;
            for (ComponentId id : include)
                if (m.test(id)) values[reg.componentInfo(id).name] = componentJson(*w, e, id);
            o["values"] = values;
        }
        out.push_back(o);
        last = e.index;
    }
    Json::object result{{"entities", out}, {"count", int(out.size())}};
    if (more) result["next"] = double(last);
    // near uses the spatial grid of the last step: entities spawned or moved since are found after the next step
    if (isNear) result["grid_step"] = double(w->stepIndex());
    return ApiResult::success(result);
}

ApiResult WorldApi::inspect(const Json& a)
{
    std::string error;
    Entity e;
    if (!aliveEntity(*w, a["entity"], e, error)) return ApiResult::failure(error);
    const Registry& reg = w->registry();
    ComponentMask only;
    bool filtered = false;
    if (!a["components"].is_null())
    {
        if (!a["components"].is_array()) return ApiResult::failure("components: expected an array of names");
        for (const Json& n : a["components"].array_items())
        {
            ComponentId id;
            if (!componentId(reg, n, id, error)) return ApiResult::failure(error);
            only.set(id);
            filtered = true;
        }
    }
    Json::object components;
    const ComponentMask& m = reg.entityMask(e);
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!m.test(ComponentId(id)) || (filtered && !only.test(ComponentId(id)))) continue;
        components[reg.componentInfo(ComponentId(id)).name] = componentJson(*w, e, ComponentId(id));
    }
    const uint8_t flags = reg.flags(e);
    Json::array children;
    for (Entity c : childrenOf(*w, e)) children.push_back(entityJson(c));
    Json::object o{{"entity", entityJson(e)},
                   {"flags", Json::object{{"simulated", bool(flags & Registry::kSimulated)},
                                          {"world_state", bool(flags & Registry::kWorldState)},
                                          {"space_state", bool(flags & Registry::kSpaceState)}}},
                   {"children", children},
                   {"components", components}};
    if (const char* c = className(*w, e)) o["class"] = c;
    if (const Parent* p = w->tryRead<Parent>(e)) o["parent"] = entityJson(p->entity);
    return ApiResult::success(o);
}

ApiResult WorldApi::getGlobal(const Json& a)
{
    std::string error;
    ComponentId id;
    if (!componentId(w->registry(), a["component"], id, error)) return ApiResult::failure(error);
    Entity holder;
    if (a["space"].is_null())
    {
        if (!w->registry().isGlobal(id))
            return ApiResult::failure(a["component"].string_value() + " is not a global (space state needs space)");
        holder = w->worldStateEntity();
    }
    else
    {
        int64_t space = 0;
        if (!intArg(a, "space", 0, 65535, 0, space, error)) return ApiResult::failure(error);
        if (!w->registry().isSpaceState(id))
            return ApiResult::failure(a["component"].string_value() + " is not space state");
        holder = w->spaceStateEntity(SpaceId(space));
        if (!holder.valid()) return ApiResult::failure("the space " + std::to_string(space) + " is not loaded");
    }
    return ApiResult::success(Json::object{{"entity", entityJson(holder)}, {"values", componentJson(*w, holder, id)}});
}

ApiResult WorldApi::getProfile(const Json& a)
{
    std::string error;
    int64_t steps = 0;
    if (!intArg(a, "steps", 0, 1000000, 0, steps, error)) return ApiResult::failure(error);
    Json report = w->profiler().report(int(steps));
    if (a["scopes"].is_null()) return ApiResult::success(report);
    if (!a["scopes"].is_array()) return ApiResult::failure("scopes: expected an array of path prefixes");
    std::vector<std::string> prefixes;
    for (const Json& p : a["scopes"].array_items())
    {
        if (!p.is_string()) return ApiResult::failure("scopes: expected strings");
        prefixes.push_back(p.string_value());
    }
    Json::array kept;
    for (const Json& s : report["scopes"].array_items())
        for (const std::string& prefix : prefixes)
            if (s["scope"].string_value().compare(0, prefix.size(), prefix) == 0)
            {
                kept.push_back(s);
                break;
            }
    Json::object o = report.object_items();
    o["scopes"]    = kept;
    return ApiResult::success(o);
}

ApiResult WorldApi::logs(const Json& a)
{
    std::string error;
    int64_t since = 0;
    if (!intArg(a, "since", 0, int64_t(1) << 53, 0, since, error)) return ApiResult::failure(error);
    if (!a["category"].is_null() && !a["category"].is_string()) return ApiResult::failure("category: a string");
    const std::string category = a["category"].string_value();
    const auto lines           = engineLog().since(uint64_t(since));
    Json::array out;
    for (const LogRing::Line& l : lines)
        if (category.empty() || l.category == category)
            out.push_back(Json::object{{"seq", double(l.sequence)}, {"category", l.category}, {"message", l.message}});
    // At most the newest 500; truncated: older matching lines exist (call again with since = first - 1 is not
    // possible: the ring keeps only the newest lines; narrow the category instead)
    const bool truncated =
        out.size() > 500 || (!lines.empty() && since > 0 && lines.front().sequence > uint64_t(since) + 1);
    if (out.size() > 500) out.erase(out.begin(), out.end() - 500);
    Json::object result;
    if (out.empty() && !category.empty())
    {
        // A hint: the categories of the kept lines
        std::vector<std::string> categories;
        for (const LogRing::Line& l : lines)
            if (std::find(categories.begin(), categories.end(), l.category) == categories.end())
                categories.push_back(l.category);
        Json::array known;
        for (auto& c : categories) known.push_back(c);
        result["categories"] = known;
    }
    result["lines"]     = out;
    result["first"]     = out.empty() ? Json() : out.front()["seq"];
    result["last"]      = double(engineLog().lastSequence());
    result["truncated"] = truncated;
    return ApiResult::success(result);
}

// ---- changes ------------------------------------------------------------------------------------------------------

void WorldApi::drainChanges()
{
    const uint64_t stepIndex = w->stepIndex();
    w->structuralLog().consume(consumer, [&](const StructuralLog::Record& r) { journalRecord(r, stepIndex); });
}

void WorldApi::trackMoves()
{
    // The entities whose Transform changed in any phase of the step that just ended, with their handle (a reused slot
    // is not reported as moved)
    const uint64_t stepIndex = w->stepIndex();
    // The handles were taken when the step ended (a listener before this one may have reused a slot since)
    for (Entity e : w->changedDuringLastStep())
    {
        if (e.index >= lastMoved.size()) lastMoved.resize(size_t(e.index) + 1);
        MoveRecord& r = lastMoved[e.index];
        r.previous    = r.entity == e ? r.step : 0;  // a new entity in the slot starts fresh
        r.entity      = e;
        r.step        = stepIndex;
    }
}

ApiResult WorldApi::changesSince(const Json& a)
{
    std::string error;
    bool includeMoved = false;
    int64_t limit     = 0;
    if (!boolArg(a, "include_moved", false, includeMoved, error) || !intArg(a, "limit", 1, 1000, 200, limit, error))
        return ApiResult::failure(error);
    // Changes made between steps (by the API or setup code) are in the log, not yet in the journal
    drainChanges();
    const uint64_t head = nextSeq - 1, now = w->stepIndex();
    Cursor c{epoch, 0, now, 0, 0};
    bool fromStart = false;
    const Json& j  = a["cursor"];
    if (j.is_null() || (j.is_number() && j.number_value() == 0))
    {
        // From the oldest kept change (moves: since tracking started)
        fromStart = true;
        c.seq     = journal.empty() ? head : journal.front().seq - 1;
        c.from    = movesTracked ? movesSince : now;
    }
    else if (j.is_string() && j.string_value() == "now")
    {
        c.seq = head;  // only what happens from now on
    }
    else if (!j.is_string() || !parseCursor(j.string_value(), c))
        return ApiResult::failure("cursor: the value returned by the previous changes_since, 0 or \"now\"");
    else if (c.epoch != epoch || c.seq > head)
        return ApiResult::failure("cursor: from another session or world; start again with 0 or \"now\"");
    // A cursor from before the last load: valid, but its move window refers to the replaced world
    const bool reloaded = !fromStart && lastLoadSeq > 0 && c.seq < lastLoadSeq;
    if (reloaded)
    {
        c.from  = movesTracked ? movesSince : now;
        c.to    = 0;
        c.index = 0;
    }
    else if (c.from > now || c.to > now)
        return ApiResult::failure("cursor: from another session or world; start again with 0 or \"now\"");
    // Changes older than the cursor's successor were dropped (the journal keeps the newest 100000)
    const bool truncated = !fromStart && !journal.empty() && journal.front().seq > c.seq + 1;

    Json::array changes;
    uint64_t last = c.seq;
    for (const Change& ch : journal)
    {
        if (ch.seq <= c.seq) continue;
        if (changes.size() == size_t(limit)) break;
        Json::object o{{"seq", double(ch.seq)}, {"step", double(ch.step)}, {"kind", ch.kind}};
        if (ch.entity.valid()) o["entity"] = entityJson(ch.entity);
        if (std::strcmp(ch.kind, "world_loaded") == 0)
        {
            o["from_step"]        = double(ch.fromStep);
            o["world_generation"] = double(ch.generation);
        }
        // The name as registered now (after a load into another build the ID may name another component)
        if (ch.component >= 0 && w->registry().registered(ComponentId(ch.component)))
            o["component"] = w->registry().componentInfo(ComponentId(ch.component)).name;
        changes.push_back(o);
        last = ch.seq;
    }
    const bool more = !journal.empty() && journal.back().seq > last;
    Json::object result{{"changes", changes}, {"more", more}, {"truncated", truncated}};
    if (reloaded) result["reloaded"] = true;
    // Without include_moved the move window is carried along unchanged
    Cursor next = c;
    next.epoch  = epoch;
    next.seq    = last;

    if (includeMoved)
    {
        if (!movesTracked)
        {
            // Tracking starts now: moves of the following steps are reported
            movesTracked = true;
            movesSince   = now;
            w->retainStepChanges();
            result["moves_tracked_from_step"] = double(movesSince);
            result["moved"]                   = Json::array{};
            next.from                         = now;
            next.to                           = 0;
            next.index                        = 0;
        }
        else
        {
            // The window (from, to]: fixed while it is paged, so later moves wait for the next window (an entity
            // that moves again after `to` is reported there, once)
            const uint64_t from = c.from;
            const uint64_t to   = c.to > 0 ? c.to : now;
            // The window's members: a snapshot taken when it opened (paging reads it, so later moves cannot push an
            // entity out of the window); rebuilt from the records if it was dropped
            auto snapshot = std::find_if(windows.begin(), windows.end(),
                                         [&](const WindowSnapshot& s) { return s.from == from && s.to == to; });
            // A continued window whose snapshot was dropped (more than 16 windows paged at once): rebuilt from the
            // records, which may miss entities that moved twice since: reported as moves_truncated
            const bool evicted = c.to > 0 && snapshot == windows.end();
            if (c.to == 0 || snapshot == windows.end())
            {
                WindowSnapshot s{from, to, {}, evicted};
                for (uint32_t i = 1; i < lastMoved.size(); ++i)
                    if (lastMoved[i].in(from, to)) s.members.push_back(lastMoved[i].entity);
                if (snapshot != windows.end()) windows.erase(snapshot);
                windows.push_back(std::move(s));
                if (windows.size() > 16) windows.pop_front();
                snapshot = windows.end() - 1;
            }
            // Every cursor continuing a rebuilt window learns that it may miss entities
            if (snapshot->truncated) result["moves_truncated"] = true;
            const uint32_t after = c.to > 0 ? c.index : 0;
            Json::array moved;
            bool movedMore     = false;
            uint32_t lastIndex = 0;
            for (Entity e : snapshot->members)
            {
                if (e.index <= after || !w->alive(e)) continue;
                if (moved.size() == size_t(limit))
                {
                    movedMore = true;
                    break;
                }
                moved.push_back(entityJson(e));
                lastIndex = e.index;
            }
            // Kept (other cursors may page the same window); the newest 16 windows stay
            result["moved"] = moved;
            if (movedMore)
            {
                result["moved_more"] = true;
                next.from            = from;
                next.to              = to;
                next.index           = lastIndex;
            }
            else
            {
                next.from  = to;
                next.to    = 0;
                next.index = 0;
            }
        }
    }
    result["cursor"] = makeCursor(next);
    result["step"]   = double(now);
    return ApiResult::success(result);
}

// ---- edits --------------------------------------------------------------------------------------------------------

ApiResult WorldApi::setComponent(const Json& a)
{
    std::string error;
    Entity e;
    ComponentId id;
    if (!aliveEntity(*w, a["entity"], e, error) || !componentId(w->registry(), a["component"], id, error))
        return ApiResult::failure(error);
    const Json& values = a["values"];
    if (!values.is_object()) return ApiResult::failure("values: expected an object of fields");
    if (id == Transform::id)
    {
        Json::object args{{"entity", entityJson(e)}};
        for (auto& kv : values.object_items())
        {
            if (kv.first != "position" && kv.first != "rotation" && kv.first != "scale")
                return ApiResult::failure("Transform." + kv.first + ": only position, rotation and scale");
            args[kv.first] = kv.second;
        }
        const Json previous = undoJson(id, componentJson(*w, e, id));
        ApiResult r         = setTransform(args);
        if (!r.ok) return r;
        return ApiResult::success(Json::object{{"values", componentJson(*w, e, id)}, {"previous", previous}});
    }
    if (!w->registry().entityMask(e).test(id))
        return ApiResult::failure("the entity has no " + std::string(w->registry().componentInfo(id).name));
    if (id == Persistent::id) return ApiResult::failure("Persistent: assigned by makePersistent (read-only)");
    // The previous values, writable fields only (set_component accepts them as an undo)
    const Json previous = undoJson(id, componentJson(*w, e, id));
    if (id == Parent::id)
    {
        Parent p = w->read<Parent>(e);
        if (!parentFromJson(*w, e, values, p, error)) return ApiResult::failure(error);
        w->remove<Parent>(e);
        w->add<Parent>(e, p);
        return ApiResult::success(Json::object{{"values", componentJson(*w, e, id)}, {"previous", previous}});
    }
    PoolBase* pool = w->registry().poolById(id);
    if (id == Behaviour::id)
    {
        // classId is fixed; the tick times must be finite and >= 0; the interval is derived again afterwards
        if (!values["tickInterval"].is_null())
            return ApiResult::failure(
                "Behaviour.tickInterval is derived from the rings around the players (set "
                "alwaysTickEveryStep, or change the class's tick policy)");
        Behaviour check = Behaviour();
        check.classId   = w->read<Behaviour>(e).classId;
        if (!fromJson(values, check, &error, true)) return ApiResult::failure(error);
        if (check.classId != w->read<Behaviour>(e).classId) return ApiResult::failure("Behaviour.classId is fixed");
        for (double t : {check.tickInterval, check.sinceTick, check.tickOffset})
            if (!(t >= 0) || !std::isfinite(t)) return ApiResult::failure("Behaviour: tick times are finite and >= 0");
    }
    if (!pool->mergeJson(e.index, values, error)) return ApiResult::failure(error);
    if (id == Behaviour::id) assignTickInterval(*w, e, w->get<Behaviour>(e), false);
    return ApiResult::success(Json::object{{"values", componentJson(*w, e, id)}, {"previous", previous}});
}

ApiResult WorldApi::addComponent(const Json& a)
{
    std::string error;
    Entity e;
    ComponentId id;
    if (!aliveEntity(*w, a["entity"], e, error) || !componentId(w->registry(), a["component"], id, error))
        return ApiResult::failure(error);
    if (w->registry().entityMask(e).test(id))
        return ApiResult::failure("the entity already has " + std::string(w->registry().componentInfo(id).name) +
                                  " (use set_component)");
    const Json& values = a["values"];
    if (!values.is_null() && !values.is_object()) return ApiResult::failure("values: expected an object of fields");
    if (!checkComponentValues(*w, e, id, values, error)) return ApiResult::failure(error);
    if (!addComponentValues(*w, e, id, values, error)) return ApiResult::failure(error);
    return ApiResult::success(componentJson(*w, e, id));
}

ApiResult WorldApi::removeComponent(const Json& a)
{
    std::string error;
    Entity e;
    ComponentId id;
    if (!aliveEntity(*w, a["entity"], e, error) || !componentId(w->registry(), a["component"], id, error))
        return ApiResult::failure(error);
    const std::string special = specialComponent(id, false);
    if (!special.empty()) return ApiResult::failure(special);
    if (!w->registry().entityMask(e).test(id))
        return ApiResult::failure("the entity has no " + std::string(w->registry().componentInfo(id).name));
    if (stateEntity(w->registry(), e) && (w->registry().isGlobal(id) || w->registry().isSpaceState(id)))
        return ApiResult::failure("world-state components cannot be removed");
    w->removeByIdNow(e, id);
    return ApiResult::success(Json::object{{"removed", w->registry().componentInfo(id).name}});
}

ApiResult WorldApi::setTransform(const Json& a)
{
    std::string error;
    Entity e;
    if (!aliveEntity(*w, a["entity"], e, error)) return ApiResult::failure(error);
    if (w->has<Parent>(e))
        return ApiResult::failure(
            "the entity is a child: its Transform follows its parent (change its LocalTransform "
            "or remove its Parent)");
    Transform& t  = w->get<Transform>(e);
    Vec3 position = t.position(), scale = t.scale();
    Quat rotation = t.rotation();
    if (!a["position"].is_null() && !readVec3(a["position"], position, error, "position"))
        return ApiResult::failure(error);
    if (!a["scale"].is_null() && !readVec3(a["scale"], scale, error, "scale")) return ApiResult::failure(error);
    if (!a["rotation"].is_null())
    {
        JsonReader r(emptyObject(), true);
        Quat q = rotation;
        r.read(a["rotation"], q, "rotation");
        if (!r.ok()) return ApiResult::failure(r.errorMessage());
        const double n = std::sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
        if (!(n > 1e-12) || !std::isfinite(n)) return ApiResult::failure("rotation: not a valid quaternion");
        rotation = Quat(q.w() / n, q.x() / n, q.y() / n, q.z() / n);
    }
    bool teleport = true;
    if (!boolArg(a, "teleport", true, teleport, error)) return ApiResult::failure(error);
    if (teleport)
        t.teleport(position, rotation);
    else
    {
        t.setPosition(position);
        t.setRotation(rotation);
    }
    t.setScale(scale);
    return ApiResult::success(toJson(t));
}

ApiResult WorldApi::spawn(const Json& a)
{
    std::string error;
    Vec3 position = Vec3::Zero(), scale = Vec3::Ones();
    Quat rotation = Quat::Identity();
    if (a["position"].is_null()) return ApiResult::failure("position: missing");
    if (!readVec3(a["position"], position, error, "position")) return ApiResult::failure(error);
    if (!a["scale"].is_null() && !readVec3(a["scale"], scale, error, "scale")) return ApiResult::failure(error);
    if (!a["rotation"].is_null())
    {
        JsonReader r(emptyObject(), true, true);
        r.read(a["rotation"], rotation, "rotation");
        if (!r.ok()) return ApiResult::failure(r.errorMessage());
    }
    int64_t space = 0;
    if (!intArg(a, "space", 0, 0, 0, space, error)) return ApiResult::failure("space: only space 0 exists (S21)");
    const ClassList::ClassInfo* info = nullptr;
    if (!a["class"].is_null())
    {
        if (!a["class"].is_string()) return ApiResult::failure("class: expected a class name");
        info = w->classList().find(a["class"].string_value());
        if (!info) return ApiResult::failure("unknown class \"" + a["class"].string_value() + "\"");
    }
    // Every initial component is checked before anything is created
    std::vector<std::pair<ComponentId, Json>> initial;
    if (!a["components"].is_null())
    {
        if (!a["components"].is_object()) return ApiResult::failure("components: expected {name: values}");
        for (auto& kv : a["components"].object_items())
        {
            ComponentId id;
            if (!componentId(w->registry(), Json(kv.first), id, error)) return ApiResult::failure(error);
            if (!kv.second.is_null() && !kv.second.is_object())
                return ApiResult::failure(kv.first + ": expected an object of fields");
            if (!checkComponentValues(*w, Entity{}, id, kv.second, error)) return ApiResult::failure(error);
            initial.push_back({id, kv.second});
        }
    }
    Entity e;
    if (info)
        e = w->spawnByClass(*info, SpaceId(space), position, {});
    else
    {
        e = w->create();
        w->get<Transform>(e).setPosition(position);
    }
    if (!w->alive(e)) return ApiResult::success(Json::object{{"entity", entityJson(e)}, {"alive", false}});
    // Only the given ones: a class may have set its own rotation or scale in configure or onSpawn
    Transform& t = w->get<Transform>(e);
    if (!a["rotation"].is_null()) t.setRotation(rotation);
    if (!a["scale"].is_null()) t.setScale(scale);
    // After the class's configure and onSpawn: a component the object added itself is merged with the given values.
    // A component that cannot be applied any more (e.g. onSpawn destroyed the Parent's entity) is skipped and reported.
    Json::array skipped;
    for (auto& c : initial)
    {
        if (!w->alive(e)) break;
        bool ok = true;
        error.clear();
        if (w->registry().entityMask(e).test(c.first))
        {
            if (c.first == Parent::id)
            {
                Parent p = w->read<Parent>(e);
                ok       = parentFromJson(*w, e, c.second, p, error);
                if (ok)
                {
                    w->remove<Parent>(e);
                    w->add<Parent>(e, p);
                }
            }
            else if (!c.second.is_null())
                ok = w->registry().poolById(c.first)->mergeJson(e.index, c.second, error);
        }
        else
            ok = addComponentValues(*w, e, c.first, c.second, error);
        if (!ok)
            skipped.push_back(Json::object{{"component", w->registry().componentInfo(c.first).name}, {"error", error}});
    }
    Json::object result{{"entity", entityJson(e)}, {"alive", w->alive(e)}};
    if (!skipped.empty()) result["skipped"] = skipped;
    return ApiResult::success(result);
}

ApiResult WorldApi::destroy(const Json& a)
{
    std::string error;
    Entity e;
    if (!aliveEntity(*w, a["entity"], e, error)) return ApiResult::failure(error);
    if (stateEntity(w->registry(), e)) return ApiResult::failure("world-state entities cannot be destroyed");
    // The entity and its descendants (reported as destroyed if gone afterwards)
    std::vector<Entity> group{e};
    for (size_t i = 0; i < group.size(); ++i)
        for (Entity c : childrenOf(*w, group[i])) group.push_back(c);
    w->destroy(e);
    Json::array destroyed;
    for (Entity x : group)
        if (!w->alive(x)) destroyed.push_back(entityJson(x));
    return ApiResult::success(Json::object{{"destroyed", destroyed}});
}

ApiResult WorldApi::setGlobal(const Json& a)
{
    if (!a["values"].is_object()) return ApiResult::failure("values: expected an object of fields");
    Json::object readArgs{{"component", a["component"]}};
    if (!a["space"].is_null()) readArgs["space"] = a["space"];
    ApiResult holder = getGlobal(readArgs);
    if (!holder.ok) return holder;
    Entity e;
    std::string error;
    ComponentId id;
    parseEntity(holder.value["entity"], e, error);
    componentId(w->registry(), a["component"], id, error);
    const Json previous = componentJson(*w, e, id);
    if (!w->registry().poolById(id)->mergeJson(e.index, a["values"], error)) return ApiResult::failure(error);
    return ApiResult::success(Json::object{{"values", componentJson(*w, e, id)}, {"previous", previous}});
}

ApiResult WorldApi::emitEvent(const Json& a)
{
    if (!a["event"].is_string()) return ApiResult::failure("event: expected an event name");
    EventQueueBase* q = w->events.find(a["event"].string_value());
    if (!q)
        return ApiResult::failure("unknown event \"" + a["event"].string_value() + "\" (describe_world lists them)");
    std::string error;
    const Json& values = a["values"].is_null() ? emptyObject() : a["values"];
    if (!q->emitJson(values, error)) return ApiResult::failure(error);
    return ApiResult::success(Json::object{{"emitted", q->name}, {"readable_in_step", double(w->stepIndex() + 1)}});
}

ApiResult WorldApi::queueAction(const Json& a)
{
    std::string error;
    Entity p;
    if (!aliveEntity(*w, a["player"], p, error)) return ApiResult::failure("player: " + error);
    if (!w->has<Player>(p)) return ApiResult::failure("player: the entity has no Player component");
    PlayerAction action;
    action.player = p;
    int64_t code  = 0;
    if (a["action"].is_null()) return ApiResult::failure("action: missing");
    if (!intArg(a, "action", 0, 0xFFFFFFFFll, 0, code, error)) return ApiResult::failure(error);
    action.action = uint32_t(code);
    if (!a["aim"].is_null() && !readVec3(a["aim"], action.aim, error, "aim")) return ApiResult::failure(error);
    w->queueAction(action);
    return ApiResult::success(Json::object{{"queued", true}, {"readable_in_step", double(w->stepIndex() + 1)}});
}

ApiResult WorldApi::step(const Json& a)
{
    if (!w->isPaused()) return ApiResult::failure("step: pause the world first");
    std::string error;
    int64_t n = 1;
    if (!intArg(a, "n", 1, 100000, 1, n, error)) return ApiResult::failure(error);
    // A wall-clock budget: a long request answers before the tool call times out
    // Inside apply the batch's deadline applies (one budget for the whole request)
    const auto deadline = inBatch ? batchDeadline : budgetDeadline();
    int64_t ran         = 0;
    for (; ran < n; ++ran)
    {
        if (std::chrono::steady_clock::now() > deadline) break;
        w->step();
    }
    return ApiResult::success(Json::object{{"step", double(w->stepIndex())},
                                           {"time", w->time()},
                                           {"ran", double(ran)},
                                           {"requested", double(n)},
                                           {"stopped_early", ran < n}});
}

ApiResult WorldApi::timeScale(const Json& a)
{
    const double scale = a["scale"].number_value();
    if (!a["scale"].is_number() || !(scale >= 0 && scale <= 1000)) return ApiResult::failure("scale: 0..1000");
    w->setTimeScale(a["scale"].number_value());
    return ApiResult::success(Json::object{{"time_scale", w->timeScale()}});
}

// ---- snapshots and data (S11) ------------------------------------------------------------------------------------

namespace
{
constexpr size_t kMaxSnapshots     = 16;
constexpr size_t kMaxSnapshotBytes = size_t(256) << 20;

bool snapshotName(const Json& a, std::string& name, std::string& error)
{
    if (!a["name"].is_string() || a["name"].string_value().empty() || a["name"].string_value().size() > 128)
    {
        error = "name: a non-empty string (at most 128 characters)";
        return false;
    }
    name = a["name"].string_value();
    return true;
}

std::string hashHex(uint64_t h)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
}

// A plain file name: letters, digits, '.', '_', '-' (no paths, no ".."), not starting with '.' (the temporary files of
// exports), not a Windows device name (CON, NUL.bin)
bool plainFileName(const std::string& f)
{
    if (f.empty() || f.size() > 128 || f[0] == '.' || f.find("..") != std::string::npos) return false;
    for (char c : f)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
              c == '-'))
            return false;
    std::string stem = f.substr(0, f.find('.'));
    for (char& c : stem) c = char(std::toupper((unsigned char)c));
    static const char* const kDevices[] = {"CON", "PRN", "AUX", "NUL"};
    for (const char* d : kDevices)
        if (stem == d) return false;
    if (stem.size() == 4 && (stem.compare(0, 3, "COM") == 0 || stem.compare(0, 3, "LPT") == 0) && stem[3] >= '0' &&
        stem[3] <= '9')
        return false;
    return true;
}
}  // namespace

Json WorldApi::removedJson(const LoadResult& r)
{
    Json::array a;
    for (Entity e : r.removed) a.push_back(entityJson(e));
    return a;
}

Json WorldApi::listenerErrorsJson(const LoadResult& r)
{
    Json::array a;
    for (const std::string& e : r.listenerErrors) a.push_back(e);
    return a;
}

Json WorldApi::hookErrorsJson(const LoadResult& r)
{
    Json::array a;
    for (const LoadResult::HookError& e : r.hookErrors)
        a.push_back(Json::object{{"entity", entityJson(e.entity)}, {"class", e.className}, {"message", e.message}});
    return a;
}

ApiResult WorldApi::saveSnapshot(const Json& a)
{
    std::string name, error;
    bool overwrite = false;
    if (!snapshotName(a, name, error) || !boolArg(a, "overwrite", false, overwrite, error))
        return ApiResult::failure(error);
    auto existing = snapshots.find(name);
    if (existing != snapshots.end() && !overwrite)
        return ApiResult::failure("save_snapshot: \"" + name + "\" exists (overwrite: true replaces it)");
    if (!w->canSave()) return ApiResult::failure("save_snapshot: a save only between steps, not inside a game hook");
    Snapshot s;
    s.bytes      = w->save();
    s.step       = w->stepIndex();
    s.hash       = w->hash();
    size_t total = s.bytes.size();
    for (const auto& kv : snapshots)
        if (kv.first != name) total += kv.second.bytes.size();
    if (existing == snapshots.end() && snapshots.size() >= kMaxSnapshots)
        return ApiResult::failure("save_snapshot: at most " + std::to_string(kMaxSnapshots) +
                                  " snapshots (delete_snapshot frees one)");
    if (total > kMaxSnapshotBytes)
        return ApiResult::failure(
            "save_snapshot: the snapshots would use more than 256 MB (delete_snapshot frees "
            "memory)");
    Json result = Json::object{
        {"name", name}, {"bytes", double(s.bytes.size())}, {"step", double(s.step)}, {"hash", hashHex(s.hash)}};
    s.revision      = nextSnapshotRevision++;
    snapshots[name] = std::move(s);
    return ApiResult::success(result);
}

ApiResult WorldApi::loadSnapshot(const Json& a)
{
    std::string name, error;
    if (!snapshotName(a, name, error)) return ApiResult::failure(error);
    if (!a["paused"].is_null() && !a["paused"].is_bool()) return ApiResult::failure("paused: expected a bool");
    auto it = snapshots.find(name);
    if (it == snapshots.end()) return ApiResult::failure("load_snapshot: no snapshot \"" + name + "\"");
    // A copy: a load listener may change the snapshots (not the API's own, but be safe)
    const std::vector<uint8_t> bytes = it->second.bytes;
    const uint64_t revision          = it->second.revision;
    World* const destination         = w;
    const LoadResult r               = destination->tryLoad(bytes);
    if (!r.committed) return ApiResult::failure("load_snapshot: " + r.error + " (the world is unchanged)");
    // A load listener rebound the API to another world: the snapshot is in the previous world; neither paused nor the
    // results of the new world apply to it
    if (w != destination)
        return ApiResult::failure("load_snapshot: \"" + name +
                                  "\" was loaded into the previous world, but a load listener rebound the API to "
                                  "another world (paused was not applied)");
    if (a["paused"].is_bool()) w->setPaused(a["paused"].bool_value());
    // An imported snapshot learns its hash (the snapshot describes exactly this world unless a hook failed), unless a
    // listener replaced it under the same name
    if (r.ok())
    {
        auto again = snapshots.find(name);
        if (again != snapshots.end() && again->second.revision == revision && !again->second.hasHash)
        {
            again->second.hash    = r.hash;  // before the load listeners could change the world
            again->second.hasHash = true;
        }
    }
    return ApiResult::success(Json::object{{"step", double(w->stepIndex())},
                                           {"hash", hashHex(w->hash())},
                                           {"entities", double(w->registry().aliveCount())},
                                           {"paused", w->isPaused()},
                                           {"world_generation", double(generation)},
                                           {"hook_errors", hookErrorsJson(r)},
                                           {"removed", removedJson(r)},
                                           {"listener_errors", listenerErrorsJson(r)}});
}

ApiResult WorldApi::listSnapshots(const Json&)
{
    Json::array list;
    for (const auto& kv : snapshots)
        list.push_back(Json::object{{"name", kv.first},
                                    {"bytes", double(kv.second.bytes.size())},
                                    {"step", double(kv.second.step)},
                                    {"hash", kv.second.hasHash ? Json(hashHex(kv.second.hash)) : Json()}});
    return ApiResult::success(Json::object{{"snapshots", list}});
}

ApiResult WorldApi::deleteSnapshot(const Json& a)
{
    std::string name, error;
    if (!snapshotName(a, name, error)) return ApiResult::failure(error);
    if (!snapshots.erase(name)) return ApiResult::failure("delete_snapshot: no snapshot \"" + name + "\"");
    return ApiResult::success(Json::object{{"deleted", name}});
}

ApiResult WorldApi::diffSnapshot(const Json& a)
{
    std::string name, error;
    if (!snapshotName(a, name, error)) return ApiResult::failure(error);
    auto it = snapshots.find(name);
    if (it == snapshots.end()) return ApiResult::failure("diff_snapshot: no snapshot \"" + name + "\"");
    DiffFilter filter;
    int64_t limit = 0, after = 0;
    if (!intArg(a, "limit", 1, 1000, 20, limit, error) || !intArg(a, "after", 0, int64_t(1) << 40, 0, after, error))
        return ApiResult::failure(error);
    filter.limit = size_t(limit);
    filter.after = size_t(after);
    if (!a["entity"].is_null())
    {
        Entity e;
        if (!parseEntity(a["entity"], e, error)) return ApiResult::failure(error);
        filter.hasEntity        = true;
        filter.entityIndex      = e.index;
        filter.hasGeneration    = true;
        filter.entityGeneration = e.generation;
    }
    if (!a["components"].is_null())
    {
        if (!a["components"].is_array()) return ApiResult::failure("components: expected an array of names");
        for (const Json& n : a["components"].array_items())
        {
            if (!n.is_string()) return ApiResult::failure("components: expected an array of names");
            // A typo must not silently match nothing
            if (w->registry().findComponent(n.string_value()) < 0 && n.string_value() != Transform::name)
                return ApiResult::failure("components: no component \"" + n.string_value() + "\"");
            filter.components.push_back(n.string_value());
        }
    }
    std::vector<uint8_t> other;
    std::string against = "live world";
    if (!a["against"].is_null())
    {
        if (!a["against"].is_string()) return ApiResult::failure("against: a snapshot name");
        auto o = snapshots.find(a["against"].string_value());
        if (o == snapshots.end())
            return ApiResult::failure("diff_snapshot: no snapshot \"" + a["against"].string_value() + "\"");
        other   = o->second.bytes;
        against = o->first;
    }
    else
    {
        if (!w->canSave())
            return ApiResult::failure(
                "diff_snapshot: the live world is saved only between steps, not inside a game hook");
        other = w->save();
    }
    const SaveDiff d = diffSaves(it->second.bytes, other, filter);
    if (!d.error.empty()) return ApiResult::failure("diff_snapshot: " + d.error);
    Json::array diffs;
    for (const SaveDifference& x : d.differences) diffs.push_back(x.toJson());
    Json::object result{
        {"a", name}, {"b", against}, {"summary", d.summary}, {"differences", diffs}, {"total", double(d.total)}};
    if (filter.after + d.differences.size() < d.total) result["next"] = double(filter.after + d.differences.size());
    return ApiResult::success(result);
}

namespace
{
// Creates a new file for writing, fails (errno EEXIST) if it exists
std::FILE* openExclusive(const std::filesystem::path& p)
{
#ifdef _WIN32
    std::FILE* f = nullptr;
    if (_wfopen_s(&f, p.c_str(), L"wbx") != 0) return nullptr;
    return f;
#else
    return std::fopen(p.c_str(), "wbx");
#endif
}
}  // namespace

ApiResult WorldApi::exportSnapshot(const Json& a)
{
    std::string name, error;
    bool overwrite = false;
    if (!snapshotName(a, name, error) || !boolArg(a, "overwrite", false, overwrite, error))
        return ApiResult::failure(error);
    if (snapshotDirectory.empty()) return ApiResult::failure("export_snapshot: no snapshot directory is set");
    if (!a["file"].is_string() || !plainFileName(a["file"].string_value()))
        return ApiResult::failure("file: a plain file name (letters, digits, '.', '_', '-'; no device names)");
    auto it = snapshots.find(name);
    if (it == snapshots.end()) return ApiResult::failure("export_snapshot: no snapshot \"" + name + "\"");
    namespace fs        = std::filesystem;
    const fs::path path = fs::path(snapshotDirectory) / a["file"].string_value();
    std::error_code ec;
    if (fs::exists(path, ec) && !overwrite)
        return ApiResult::failure("export_snapshot: " + path.string() + " exists (overwrite: true replaces it)");
    // A temporary file of its own in the directory, created exclusively (never a user's file or another export's
    // temporary, also one left over by an earlier process): written completely, then renamed into place, so a failed
    // write never leaves a truncated snapshot file; only the file created here is removed on failure
    static uint64_t exportCounter = 0;
    fs::path partial;
    std::FILE* f = nullptr;
    for (int attempt = 0; attempt < 100 && !f; ++attempt)
    {
        partial = fs::path(snapshotDirectory) /
                  (".saiga-export-" + std::to_string(epoch) + "-" + std::to_string(++exportCounter) + ".partial");
        f = openExclusive(partial);
        if (!f && errno != EEXIST) break;
    }
    if (!f) return ApiResult::failure("export_snapshot: cannot create a temporary file in " + snapshotDirectory);
    const auto& bytes  = it->second.bytes;
    const bool written = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size() && std::fflush(f) == 0;
    const bool closed  = std::fclose(f) == 0;
    if (!written || !closed)
    {
        fs::remove(partial, ec);
        return ApiResult::failure("export_snapshot: cannot write " + path.string());
    }
    fs::rename(partial, path, ec);
    if (ec)
    {
        fs::remove(partial, ec);
        return ApiResult::failure("export_snapshot: cannot write " + path.string());
    }
    return ApiResult::success(Json::object{{"file", path.string()}, {"bytes", double(it->second.bytes.size())}});
}

ApiResult WorldApi::importSnapshot(const Json& a)
{
    std::string name, error;
    bool overwrite = false;
    if (!snapshotName(a, name, error) || !boolArg(a, "overwrite", false, overwrite, error))
        return ApiResult::failure(error);
    if (snapshotDirectory.empty()) return ApiResult::failure("import_snapshot: no snapshot directory is set");
    if (!a["file"].is_string() || !plainFileName(a["file"].string_value()))
        return ApiResult::failure("file: a plain file name (letters, digits, '.', '_', '-'; no device names)");
    if (snapshots.count(name) && !overwrite)
        return ApiResult::failure("import_snapshot: \"" + name + "\" exists (overwrite: true replaces it)");
    if (!snapshots.count(name) && snapshots.size() >= kMaxSnapshots)
        return ApiResult::failure("import_snapshot: at most " + std::to_string(kMaxSnapshots) + " snapshots");
    namespace fs        = std::filesystem;
    const fs::path path = fs::path(snapshotDirectory) / a["file"].string_value();
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec) return ApiResult::failure("import_snapshot: cannot read " + path.string());
    // The limit before anything is read
    size_t total = size_t(std::min<uintmax_t>(size, kMaxSnapshotBytes + 1));
    for (const auto& kv : snapshots)
        if (kv.first != name) total += kv.second.bytes.size();
    if (size > kMaxSnapshotBytes || total > kMaxSnapshotBytes)
        return ApiResult::failure("import_snapshot: the snapshots would use more than 256 MB");
    Snapshot s;
    s.bytes.resize(size_t(size));
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(s.bytes.data()), std::streamsize(s.bytes.size()));
    if (!f) return ApiResult::failure("import_snapshot: cannot read " + path.string());
    ParsedSave parsed;
    if (!parseSave(s.bytes, parsed, error)) return ApiResult::failure("import_snapshot: " + error);
    s.step          = parsed.steps;
    s.hasHash       = false;  // the hash of the world it came from is not known
    s.revision      = nextSnapshotRevision++;
    snapshots[name] = std::move(s);
    return ApiResult::success(
        Json::object{{"name", name}, {"bytes", double(size)}, {"step", double(parsed.steps)}, {"hash", Json()}});
}

ApiResult WorldApi::getData(const Json& a)
{
    if (!a["type"].is_string()) return ApiResult::failure("type: a data type name (describe_world lists them)");
    const DataTableBase* t = w->dataTables().find(a["type"].string_value());
    if (!t) return ApiResult::failure("get_data: no data type \"" + a["type"].string_value() + "\"");
    if (a["name"].is_null())
    {
        Json::array names;
        for (auto& n : t->names()) names.push_back(n);
        return ApiResult::success(Json::object{{"type", t->name}, {"file", t->file}, {"entries", names}});
    }
    if (!a["name"].is_string()) return ApiResult::failure("name: an entry name");
    const Json v = t->entryJson(a["name"].string_value());
    if (v.is_null())
        return ApiResult::failure("get_data: " + t->file + " has no entry \"" + a["name"].string_value() + "\"");
    return ApiResult::success(Json::object{{"type", t->name}, {"name", a["name"]}, {"values", v}});
}

ApiResult WorldApi::reloadData(const Json& a)
{
    std::string error;
    bool reconfigure = true;
    if (!boolArg(a, "reconfigure", true, reconfigure, error)) return ApiResult::failure(error);
    if (!w->reloadData(error)) return ApiResult::failure("reload_data: " + error + " (nothing changed)");
    Json::array types;
    double entries = 0;
    for (const auto& t : w->dataTables().tables())
    {
        types.push_back(t->name);
        entries += double(t->names().size());
    }
    Json::object result{{"types", types}, {"entries", entries}};
    if (reconfigure)
    {
        // Every game object runs configure() with the new values (a save and load of the world)
        if (!w->canSave())
            return ApiResult::failure(
                "reload_data: the data were reloaded, the reconfigure needs a save between steps, "
                "not inside a game hook");
        const LoadResult r = w->tryLoad(w->save());
        if (!r.committed)
            return ApiResult::failure("reload_data: the data were reloaded, the reconfigure failed: " + r.error);
        result["hook_errors"]      = hookErrorsJson(r);
        result["listener_errors"]  = listenerErrorsJson(r);
        result["world_generation"] = double(generation);
    }
    return ApiResult::success(result);
}

ApiResult WorldApi::apply(const Json& a)
{
    if (!a["operations"].is_array()) return ApiResult::failure("operations: expected [{op, ...arguments}]");
    // The batch is checked for known operations and the permission first
    for (const Json& op : a["operations"].array_items())
    {
        if (!op.is_object() || !op["op"].is_string()) return ApiResult::failure("each operation needs \"op\"");
        const std::string name = op["op"].string_value();
        if (name == "apply") return ApiResult::failure("apply cannot be nested");
        if (!findOperation(name)) return ApiResult::failure("unknown operation \"" + name + "\"");
        if (isEdit(name) && !allowEdits) return ApiResult::failure(name + ": edits are not allowed (read-only)");
        if (isEdit(name) && w->dispatchingListeners())
            return ApiResult::failure(name + ": not inside a step listener (nothing of the batch was applied)");
    }
    Json::array results;
    size_t index  = 0;
    inBatch       = true;
    batchDeadline = budgetDeadline();
    struct BatchEnd
    {
        bool& flag;
        ~BatchEnd() { flag = false; }
    } batchEnd{inBatch};
    for (const Json& op : a["operations"].array_items())
    {
        // The request's deadline: the remaining operations are not run (the completed results are returned)
        if (std::chrono::steady_clock::now() > batchDeadline)
        {
            ApiResult stopped = ApiResult::failure("the time budget ended before operation " + std::to_string(index) +
                                                   " (the operations before it were applied)");
            stopped.value     = Json::object{{"results", results}, {"applied", int(index)}, {"stopped_early", true}};
            return stopped;
        }
        Json::object args = op.object_items();
        args.erase("op");
        ApiResult r = call(op["op"].string_value(), Json(args));
        if (!r.ok)
        {
            ApiResult failed =
                ApiResult::failure("operation " + std::to_string(index) + " (" + op["op"].string_value() +
                                   "): " + r.error + " (the operations before it were applied)");
            failed.value = Json::object{{"results", results}, {"applied", int(index)}};
            return failed;
        }
        results.push_back(r.value);
        index++;
    }
    return ApiResult::success(Json::object{{"results", results}, {"applied", int(index)}});
}

}  // namespace Engine
}  // namespace Saiga
