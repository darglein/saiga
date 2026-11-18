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
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>
#include <map>

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
};
const Operation kOperations[] = {
    {"describe_world", false},
    {"world_status", false},
    {"list_entities", false},
    {"inspect", false},
    {"get_global", false},
    {"changes_since", false},
    {"get_profile", false},
    {"logs", false},
    {"set_component", true},
    {"add_component", true},
    {"remove_component", true},
    {"set_transform", true},
    {"spawn", true},
    {"destroy", true},
    {"set_global", true},
    {"emit_event", true},
    {"queue_action", true},
    {"pause", true},
    {"resume", true},
    {"step", true},
    {"time_scale", true},
    {"reset_profile", true},
    {"apply", true},
};

bool readVec3(const Json& j, Vec3& out, std::string& error, const char* what)
{
    JsonReader r(Json::object{}, true);
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
}  // namespace

// ---- setup --------------------------------------------------------------------------------------------------------

WorldApi::WorldApi(World& world, bool allowEdits)
    : w(world), allowEdits(allowEdits), alive(std::make_shared<bool>(true))
{
    consumer = w.structuralLog().addConsumer();
    // The journal is filled after every step, so the structural log is never held back by an idle API
    std::weak_ptr<bool> weak = alive;
    w.addStepListener(
        [this, weak]()
        {
            if (weak.expired()) return false;
            drainChanges(true);
            return true;
        });
}

WorldApi::~WorldApi()
{
    alive.reset();
    w.structuralLog().removeConsumer(consumer);
}

std::vector<std::string> WorldApi::operations() const
{
    std::vector<std::string> names;
    for (const Operation& o : kOperations)
        if (allowEdits || !o.edit) names.push_back(o.name);
    return names;
}

bool WorldApi::isEdit(const std::string& operation)
{
    for (const Operation& o : kOperations)
        if (operation == o.name) return o.edit;
    return false;
}

bool WorldApi::parseEntity(const Json& j, Entity& e, std::string& error)
{
    JsonReader r(Json::object{}, true);
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

ApiResult WorldApi::call(const std::string& operation, const Json& a)
{
    const Json args = a.is_null() ? Json(Json::object{}) : a;
    if (!args.is_object()) return ApiResult::failure("the arguments must be an object");
    bool known = false;
    for (const Operation& o : kOperations) known = known || operation == o.name;
    if (!known) return ApiResult::failure("unknown operation \"" + operation + "\"");
    if (isEdit(operation) && !allowEdits) return ApiResult::failure(operation + ": edits are not allowed");
    if (w.inStep()) return ApiResult::failure(operation + ": only between steps");
    if (isEdit(operation) && operation != "apply")
        SAIGA_LOG("api", "%s %s", operation.c_str(), truncateForLog(args.dump()).c_str());

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
    if (operation == "pause") return pause(args);
    if (operation == "resume") return resume(args);
    if (operation == "step") return step(args);
    if (operation == "time_scale") return timeScale(args);
    if (operation == "reset_profile") return resetProfile(args);
    return apply(args);
}

// ---- helpers --------------------------------------------------------------------------------------------------------

namespace
{
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

// A Parent value from JSON, checked: an alive parent, no cycle
bool parentFromJson(World& w, Entity child, const Json& values, Parent& out, std::string& error)
{
    Parent p = out;
    if (!fromJson(values, p, &error)) return false;
    if (!w.alive(p.entity))
    {
        error = "Parent: the parent entity is not alive";
        return false;
    }
    if (isSelfOrDescendant(w, child, p.entity))
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
        return parentFromJson(w, child, values.is_null() ? Json(Json::object{}) : values, p, error);
    }
    if (values.is_null()) return true;
    return w.registry().poolById(id)->checkJson(values, error);
}

// Adds a component with values (checked before)
void addComponentValues(World& w, Entity e, ComponentId id, const Json& values)
{
    std::string error;
    if (id == Parent::id)
    {
        Parent p;
        parentFromJson(w, e, values, p, error);
        w.add<Parent>(e, p);
        return;
    }
    w.addDefaultNow(e, id);
    if (!values.is_null()) w.registry().poolById(id)->mergeJson(e.index, values, error);
}
}  // namespace

// ---- reading ------------------------------------------------------------------------------------------------------

ApiResult WorldApi::describeWorld(const Json&)
{
    const Registry& reg = w.registry();
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
        if (id == Transform::id) o["note"] = "every entity has one; change it with set_transform";
        if (id == Behaviour::id) o["note"] = "added by spawn with a class; classId cannot be changed";
        if (id == Persistent::id) o["note"] = "assigned by makePersistent";
        components.push_back(o);
    }
    Json::array classes;
    for (const auto& c : w.classList().all())
        if (c.factory) classes.push_back(Json::object{{"name", c.name}, {"id", int(c.id)}});
    Json::array events;
    for (int id = 0; id < 65536; ++id)
        if (const EventQueueBase* q = w.events.find(EventId(id)))
            events.push_back(Json::object{{"name", q->name}, {"id", id}, {"fields", fieldsJson(q->fields())}});
    Json::array ops;
    for (auto& o : operations()) ops.push_back(o);
    return ApiResult::success(Json::object{{"components", components},
                                           {"classes", classes},
                                           {"events", events},
                                           {"entity_format", "{\"index\": i, \"generation\": g} or \"i:g\""},
                                           {"operations", ops}});
}

ApiResult WorldApi::worldStatus(const Json&)
{
    const Registry& reg  = w.registry();
    const size_t objects = reg.registered(Behaviour::id) ? reg.poolById(Behaviour::id)->size() : 0;
    char hash[32];
    std::snprintf(hash, sizeof(hash), "%016llx", (unsigned long long)w.hash());
    return ApiResult::success(Json::object{{"step", double(w.stepIndex())},
                                           {"time", w.time()},
                                           {"paused", w.isPaused()},
                                           {"time_scale", w.timeScale()},
                                           {"steps_per_second", w.worldConfig().stepsPerSecond},
                                           {"entities", double(reg.aliveCount())},
                                           {"game_objects", double(objects)},
                                           {"hash", hash}});
}

ApiResult WorldApi::listEntities(const Json& a)
{
    const Registry& reg = w.registry();
    std::string error;
    // Filters
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
    int classFilter = -1;
    if (!a["class"].is_null())
    {
        if (!a["class"].is_string()) return ApiResult::failure("class: expected a class name");
        const ClassList::ClassInfo* c = w.classList().find(a["class"].string_value());
        if (!c) return ApiResult::failure("unknown class \"" + a["class"].string_value() + "\"");
        classFilter = c->id;
    }
    double limitValue = a["limit"].is_null() ? 100 : a["limit"].number_value();
    if (!a["limit"].is_null() && (!a["limit"].is_number() || limitValue < 1 || limitValue > 1000))
        return ApiResult::failure("limit: 1..1000");
    const size_t limit = size_t(limitValue);
    uint32_t after     = 0;
    if (!a["after"].is_null())
    {
        if (!a["after"].is_number() || a["after"].number_value() < 0)
            return ApiResult::failure("after: an entity index");
        after = uint32_t(a["after"].number_value());
    }

    // Candidates in index order: the grid for near (positions of the last phase 6), else every slot
    std::vector<Entity> candidates;
    if (!a["near"].is_null())
    {
        const Json& near = a["near"];
        if (!near.is_object()) return ApiResult::failure("near: expected {space?, position, radius}");
        Vec3 center = Vec3::Zero();
        if (!readVec3(near["position"], center, error, "near.position")) return ApiResult::failure(error);
        if (!near["radius"].is_number() || !(near["radius"].number_value() >= 0))
            return ApiResult::failure("near.radius: a number >= 0");
        const double space = near["space"].is_null() ? 0 : near["space"].number_value();
        if (!(space >= 0 && space <= 65535)) return ApiResult::failure("near.space: a space ID");
        candidates = w.queryRadius(SpaceId(space), center, near["radius"].number_value());
    }
    else
    {
        for (uint32_t i = after + 1; i < reg.slotCount(); ++i)
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
        if (e.index <= after) continue;
        if (!reg.entityMask(e).containsAll(mask)) continue;
        const Behaviour* b = w.tryRead<Behaviour>(e);
        if (classFilter >= 0 && (!b || b->classId != classFilter)) continue;
        if (out.size() == limit)
        {
            more = true;
            break;
        }
        Json::array names;
        const ComponentMask& m = reg.entityMask(e);
        for (size_t id = 0; id < kMaxComponents; ++id)
            if (m.test(ComponentId(id))) names.push_back(reg.componentInfo(ComponentId(id)).name);
        Json::object o{
            {"entity", entityJson(e)}, {"position", toJson(w.read<Transform>(e).position())}, {"components", names}};
        if (const char* c = className(w, e)) o["class"] = c;
        out.push_back(o);
        last = e.index;
    }
    Json::object result{{"entities", out}, {"count", int(out.size())}};
    if (more) result["next"] = double(last);
    return ApiResult::success(result);
}

ApiResult WorldApi::inspect(const Json& a)
{
    std::string error;
    Entity e;
    if (!aliveEntity(w, a["entity"], e, error)) return ApiResult::failure(error);
    const Registry& reg = w.registry();
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
        components[reg.componentInfo(ComponentId(id)).name] = componentJson(w, e, ComponentId(id));
    }
    const uint8_t flags = reg.flags(e);
    Json::array children;
    for (Entity c : childrenOf(w, e)) children.push_back(entityJson(c));
    Json::object o{{"entity", entityJson(e)},
                   {"flags", Json::object{{"simulated", bool(flags & Registry::kSimulated)},
                                          {"world_state", bool(flags & Registry::kWorldState)},
                                          {"space_state", bool(flags & Registry::kSpaceState)}}},
                   {"children", children},
                   {"components", components}};
    if (const char* c = className(w, e)) o["class"] = c;
    if (const Parent* p = w.tryRead<Parent>(e)) o["parent"] = entityJson(p->entity);
    return ApiResult::success(o);
}

ApiResult WorldApi::getGlobal(const Json& a)
{
    std::string error;
    ComponentId id;
    if (!componentId(w.registry(), a["component"], id, error)) return ApiResult::failure(error);
    Entity holder;
    if (a["space"].is_null())
    {
        if (!w.registry().isGlobal(id)) return ApiResult::failure(a["component"].string_value() + " is not a global");
        holder = w.worldStateEntity();
    }
    else
    {
        if (!w.registry().isSpaceState(id))
            return ApiResult::failure(a["component"].string_value() + " is not space state");
        holder = w.spaceStateEntity(SpaceId(a["space"].number_value()));
        if (!holder.valid()) return ApiResult::failure("the space is not loaded");
    }
    return ApiResult::success(Json::object{{"entity", entityJson(holder)}, {"values", componentJson(w, holder, id)}});
}

ApiResult WorldApi::getProfile(const Json& a)
{
    const int steps = a["steps"].is_number() ? int(a["steps"].number_value()) : 0;
    return ApiResult::success(w.profiler().report(steps));
}

ApiResult WorldApi::logs(const Json& a)
{
    const uint64_t since       = a["since"].is_number() ? uint64_t(std::max(0.0, a["since"].number_value())) : 0;
    const std::string category = a["category"].is_string() ? a["category"].string_value() : "";
    Json::array lines;
    for (const LogRing::Line& l : engineLog().since(since))
        if (category.empty() || l.category == category)
            lines.push_back(
                Json::object{{"seq", double(l.sequence)}, {"category", l.category}, {"message", l.message}});
    // At most the newest 500
    if (lines.size() > 500) lines.erase(lines.begin(), lines.end() - 500);
    return ApiResult::success(Json::object{{"lines", lines}, {"last", double(engineLog().lastSequence())}});
}

// ---- changes ------------------------------------------------------------------------------------------------------

void WorldApi::drainChanges(bool moves)
{
    const uint64_t stepIndex = w.stepIndex();
    auto push                = [&](const char* kind, Entity e, int component)
    {
        journal.push_back(Change{nextSeq++, stepIndex, kind, e, component});
        if (journal.size() > kJournalCapacity) journal.pop_front();
    };
    w.structuralLog().consume(consumer,
                              [&](const StructuralLog::Record& r)
                              {
                                  if (r.component == Transform::id)
                                      push(r.added ? "created" : "destroyed", r.entity, -1);
                                  else
                                      push(r.added ? "added" : "removed", r.entity, r.component);
                              });
    if (!moves) return;
    // Moved in the step that just ended (in the changed list of its phase 6 and changed during the step)
    const Registry& reg = w.registry();
    for (uint32_t index : w.changedTransforms())
    {
        const Entity e = reg.aliveAt(index);
        if (e.valid() && w.read<Transform>(e).changedInLastStep()) push("moved", e, -1);
    }
}

ApiResult WorldApi::changesSince(const Json& a)
{
    // Changes made between steps (by the API or setup code) are in the log, not yet in the journal
    drainChanges(false);
    const uint64_t cursor = a["cursor"].is_number() ? uint64_t(std::max(0.0, a["cursor"].number_value())) : 0;
    // Older changes than the cursor's successor were dropped (the journal keeps the newest 100000)
    const bool truncated = cursor > 0 && !journal.empty() && journal.front().seq > cursor + 1;
    Json::array changes;
    uint64_t last = cursor;
    for (const Change& c : journal)
    {
        if (c.seq <= cursor) continue;
        if (changes.size() == 1000) break;
        Json::object o{
            {"seq", double(c.seq)}, {"step", double(c.step)}, {"kind", c.kind}, {"entity", entityJson(c.entity)}};
        if (c.component >= 0) o["component"] = w.registry().componentInfo(ComponentId(c.component)).name;
        changes.push_back(o);
        last = c.seq;
    }
    return ApiResult::success(Json::object{{"changes", changes},
                                           {"cursor", double(last)},
                                           {"more", !journal.empty() && journal.back().seq > last},
                                           {"truncated", truncated}});
}

// ---- edits --------------------------------------------------------------------------------------------------------

ApiResult WorldApi::setComponent(const Json& a)
{
    std::string error;
    Entity e;
    ComponentId id;
    if (!aliveEntity(w, a["entity"], e, error) || !componentId(w.registry(), a["component"], id, error))
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
        ApiResult r = setTransform(args);
        if (!r.ok) return r;
        return ApiResult::success(componentJson(w, e, id));
    }
    if (!w.registry().entityMask(e).test(id))
        return ApiResult::failure("the entity has no " + std::string(w.registry().componentInfo(id).name));
    if (id == Persistent::id) return ApiResult::failure("Persistent: assigned by makePersistent");
    if (id == Parent::id)
    {
        Parent p = w.read<Parent>(e);
        if (!parentFromJson(w, e, values, p, error)) return ApiResult::failure(error);
        w.remove<Parent>(e);
        w.add<Parent>(e, p);
        return ApiResult::success(componentJson(w, e, id));
    }
    if (id == Behaviour::id && !values["classId"].is_null() &&
        values["classId"].number_value() != double(w.read<Behaviour>(e).classId))
        return ApiResult::failure("Behaviour.classId cannot be changed");
    PoolBase* pool = w.registry().poolById(id);
    if (!pool->mergeJson(e.index, values, error)) return ApiResult::failure(error);
    return ApiResult::success(componentJson(w, e, id));
}

ApiResult WorldApi::addComponent(const Json& a)
{
    std::string error;
    Entity e;
    ComponentId id;
    if (!aliveEntity(w, a["entity"], e, error) || !componentId(w.registry(), a["component"], id, error))
        return ApiResult::failure(error);
    if (w.registry().entityMask(e).test(id))
        return ApiResult::failure("the entity already has " + std::string(w.registry().componentInfo(id).name) +
                                  " (use set_component)");
    const Json& values = a["values"];
    if (!values.is_null() && !values.is_object()) return ApiResult::failure("values: expected an object of fields");
    if (!checkComponentValues(w, e, id, values, error)) return ApiResult::failure(error);
    addComponentValues(w, e, id, values);
    return ApiResult::success(componentJson(w, e, id));
}

ApiResult WorldApi::removeComponent(const Json& a)
{
    std::string error;
    Entity e;
    ComponentId id;
    if (!aliveEntity(w, a["entity"], e, error) || !componentId(w.registry(), a["component"], id, error))
        return ApiResult::failure(error);
    const std::string special = specialComponent(id, false);
    if (!special.empty()) return ApiResult::failure(special);
    if (!w.registry().entityMask(e).test(id))
        return ApiResult::failure("the entity has no " + std::string(w.registry().componentInfo(id).name));
    if (stateEntity(w.registry(), e) && (w.registry().isGlobal(id) || w.registry().isSpaceState(id)))
        return ApiResult::failure("world-state components cannot be removed");
    w.removeByIdNow(e, id);
    return ApiResult::success(Json::object{{"removed", w.registry().componentInfo(id).name}});
}

ApiResult WorldApi::setTransform(const Json& a)
{
    std::string error;
    Entity e;
    if (!aliveEntity(w, a["entity"], e, error)) return ApiResult::failure(error);
    Transform& t  = w.get<Transform>(e);
    Vec3 position = t.position(), scale = t.scale();
    Quat rotation = t.rotation();
    if (!a["position"].is_null() && !readVec3(a["position"], position, error, "position"))
        return ApiResult::failure(error);
    if (!a["scale"].is_null() && !readVec3(a["scale"], scale, error, "scale")) return ApiResult::failure(error);
    if (!a["rotation"].is_null())
    {
        JsonReader r(Json::object{}, true);
        Quat q = rotation;
        r.read(a["rotation"], q, "rotation");
        if (!r.ok()) return ApiResult::failure(r.errorMessage());
        const double n = std::sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
        if (!(n > 1e-12) || !std::isfinite(n)) return ApiResult::failure("rotation: not a valid quaternion");
        rotation = Quat(q.w() / n, q.x() / n, q.y() / n, q.z() / n);
    }
    const bool teleport = a["teleport"].is_null() ? true : a["teleport"].bool_value();
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
    Vec3 position = Vec3::Zero();
    if (a["position"].is_null()) return ApiResult::failure("position: missing");
    if (!readVec3(a["position"], position, error, "position")) return ApiResult::failure(error);
    const double space = a["space"].is_null() ? 0 : a["space"].number_value();
    if (space != 0) return ApiResult::failure("space: only space 0 exists (before S21)");
    const ClassList::ClassInfo* info = nullptr;
    if (!a["class"].is_null())
    {
        if (!a["class"].is_string()) return ApiResult::failure("class: expected a class name");
        info = w.classList().find(a["class"].string_value());
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
            if (!componentId(w.registry(), Json(kv.first), id, error)) return ApiResult::failure(error);
            if (!kv.second.is_null() && !kv.second.is_object())
                return ApiResult::failure(kv.first + ": expected an object of fields");
            if (!checkComponentValues(w, Entity{}, id, kv.second, error)) return ApiResult::failure(error);
            initial.push_back({id, kv.second});
        }
    }
    auto addInitial = [&](Entity e)
    {
        for (auto& c : initial) addComponentValues(w, e, c.first, c.second);
    };
    Entity e;
    if (info)
        e = w.spawnByClass(*info, SpaceId(space), position, addInitial);
    else
    {
        e = w.create();
        w.get<Transform>(e).setPosition(position);
        addInitial(e);
    }
    Json::object o{{"entity", entityJson(e)}, {"alive", w.alive(e)}};
    return ApiResult::success(o);
}

ApiResult WorldApi::destroy(const Json& a)
{
    std::string error;
    Entity e;
    if (!aliveEntity(w, a["entity"], e, error)) return ApiResult::failure(error);
    if (stateEntity(w.registry(), e)) return ApiResult::failure("world-state entities cannot be destroyed");
    // The entity and its descendants (reported as destroyed if gone afterwards)
    std::vector<Entity> group{e};
    for (size_t i = 0; i < group.size(); ++i)
        for (Entity c : childrenOf(w, group[i])) group.push_back(c);
    w.destroy(e);
    Json::array destroyed;
    for (Entity x : group)
        if (!w.alive(x)) destroyed.push_back(entityJson(x));
    return ApiResult::success(Json::object{{"destroyed", destroyed}});
}

ApiResult WorldApi::setGlobal(const Json& a)
{
    ApiResult holder = getGlobal(a);
    if (!holder.ok) return holder;
    if (!a["values"].is_object()) return ApiResult::failure("values: expected an object of fields");
    Entity e;
    std::string error;
    parseEntity(holder.value["entity"], e, error);
    ComponentId id;
    componentId(w.registry(), a["component"], id, error);
    if (!w.registry().poolById(id)->mergeJson(e.index, a["values"], error)) return ApiResult::failure(error);
    return ApiResult::success(componentJson(w, e, id));
}

ApiResult WorldApi::emitEvent(const Json& a)
{
    if (!a["event"].is_string()) return ApiResult::failure("event: expected an event name");
    EventQueueBase* q = w.events.find(a["event"].string_value());
    if (!q)
        return ApiResult::failure("unknown event \"" + a["event"].string_value() + "\" (describe_world lists them)");
    std::string error;
    const Json values = a["values"].is_null() ? Json(Json::object{}) : a["values"];
    if (!q->emitJson(values, error)) return ApiResult::failure(error);
    return ApiResult::success(Json::object{{"emitted", q->name}, {"readable_in_step", double(w.stepIndex() + 1)}});
}

ApiResult WorldApi::queueAction(const Json& a)
{
    std::string error;
    Entity p;
    if (!aliveEntity(w, a["player"], p, error)) return ApiResult::failure("player: " + error);
    if (!w.has<Player>(p)) return ApiResult::failure("player: the entity has no Player component");
    PlayerAction action;
    action.player = p;
    JsonReader r(Json::object{}, true);
    if (a["action"].is_null()) return ApiResult::failure("action: missing");
    r.read(a["action"], action.action, "action");
    if (!a["aim"].is_null()) r.read(a["aim"], action.aim, "aim");
    if (!r.ok()) return ApiResult::failure(r.errorMessage());
    w.queueAction(action);
    return ApiResult::success(Json::object{{"queued", true}, {"readable_in_step", double(w.stepIndex() + 1)}});
}

ApiResult WorldApi::pause(const Json&)
{
    w.setPaused(true);
    return ApiResult::success(Json::object{{"paused", true}, {"step", double(w.stepIndex())}});
}

ApiResult WorldApi::resume(const Json&)
{
    w.setPaused(false);
    return ApiResult::success(Json::object{{"paused", false}, {"step", double(w.stepIndex())}});
}

ApiResult WorldApi::step(const Json& a)
{
    if (!w.isPaused()) return ApiResult::failure("step: pause the world first");
    const double n = a["n"].is_null() ? 1 : a["n"].number_value();
    if (!a["n"].is_null() && !a["n"].is_number()) return ApiResult::failure("n: a number of steps");
    if (!(n >= 1 && n <= 100000) || n != std::floor(n)) return ApiResult::failure("n: 1..100000");
    w.step(int(n));
    return ApiResult::success(Json::object{{"step", double(w.stepIndex())}, {"time", w.time()}});
}

ApiResult WorldApi::timeScale(const Json& a)
{
    if (!a["scale"].is_number() || !std::isfinite(a["scale"].number_value()) || a["scale"].number_value() < 0)
        return ApiResult::failure("scale: a finite number >= 0");
    w.setTimeScale(a["scale"].number_value());
    return ApiResult::success(Json::object{{"time_scale", w.timeScale()}});
}

ApiResult WorldApi::resetProfile(const Json&)
{
    w.profiler().reset();
    return ApiResult::success(Json::object{{"reset", true}});
}

ApiResult WorldApi::apply(const Json& a)
{
    if (!a["operations"].is_array()) return ApiResult::failure("operations: expected [{op, ...arguments}]");
    // The whole batch is checked for known, allowed operations first
    for (const Json& op : a["operations"].array_items())
    {
        if (!op["op"].is_string()) return ApiResult::failure("each operation needs \"op\"");
        const std::string name = op["op"].string_value();
        if (name == "apply") return ApiResult::failure("apply cannot be nested");
        bool known = false;
        for (const Operation& o : kOperations) known = known || name == o.name;
        if (!known) return ApiResult::failure("unknown operation \"" + name + "\"");
    }
    Json::array results;
    size_t index = 0;
    for (const Json& op : a["operations"].array_items())
    {
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
